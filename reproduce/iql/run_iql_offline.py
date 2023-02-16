import os
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import wandb
from tqdm import trange
from offlinerllib.utils.d4rl import get_d4rl_dataset
from offlinerllib.policy.model_free import IQLPolicy
from offlinerllib.utils.eval import eval_policy

from UtilsRL.exp import parse_args, setup
from UtilsRL.logger import CompositeLogger
import numpy as np

from typing import Callable

args = parse_args()
exp_name = "_".join([args.task, args.name, "seed"+str(args.seed)]) 
logger = CompositeLogger(log_path="./log/iql/offline", name=exp_name, loggers_config={
    "FileLogger": {"activate": not args.debug}, 
    "TensorboardLogger": {"activate": not args.debug}, 
    "WandbLogger": {"activate": not args.debug, "config": args, "settings": wandb.Settings(_disable_stats=True), **args.wandb}
})
setup(args, logger)

env, dataset = get_d4rl_dataset(args.task, normalize_obs=args.normalize_obs, normalize_reward=args.normalize_reward)
obs_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[-1]

class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)

class MLP(nn.Module):
    def __init__(
        self,
        dims,
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Callable[[], nn.Module] = None,
        squeeze_output: bool = False,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
    ):
        super().__init__()
        self.net = MLP([state_dim, *([hidden_dim] * n_hidden), act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action
        self.name="gaussian"

    def forward(self, obs: torch.Tensor) -> MultivariateNormal:
        mean = self.net(obs)
        std = torch.exp(self.log_std.clamp(-5, 2))
        scale_tril = torch.diag(std)
        return MultivariateNormal(mean, scale_tril=scale_tril)
    
    def sample(self, obs, deterministic: bool=False, *args, **kwargs):
        dist = self(obs)
        action = dist.mean if deterministic else dist.sample()
        action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()
    
    def evaluate(self, obs, action, return_dist=False, *args, **kwargs):
        dist = self.forward(obs)
        return dist.log_prob(action).unsqueeze(1), {}
        


class DeterministicPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        max_action: float,
        hidden_dim: int = 256,
        n_hidden: int = 2,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
        )
        self.max_action = max_action
        self.name="deterministic"

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
    
    def sample(self, obs, deterministic: bool=False, *args, **kwargs):
        return self(obs)


class TwinQ(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim + action_dim, *([hidden_dim] * n_hidden), 1]
        self.q1 = MLP(dims, squeeze_output=False)
        self.q2 = MLP(dims, squeeze_output=False)

    def both(
        self, state: torch.Tensor, action: torch.Tensor
    ):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def forward(self, state: torch.Tensor, action: torch.Tensor, reduce=True) -> torch.Tensor:
        if reduce:
            return torch.min(*self.both(state, action))
        else:
            return torch.stack(self.both(state, action), dim=0)


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=False)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)
    
if args.iql_deterministic: 
    actor = DeterministicPolicy(
        state_dim=obs_shape, 
        act_dim=action_shape,
        max_action=1.0, 
        hidden_dim=256, 
        n_hidden=2 
    ).to(args.device)
else:
    actor = GaussianPolicy(
        state_dim=obs_shape, 
        act_dim=action_shape, 
        max_action=1.0, 
        hidden_dim=256, 
        n_hidden=2, 
    ).to(args.device)
actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
if args.actor_opt_decay_schedule:
    actor_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, args.max_epoch * args.step_per_epoch)
else:
    actor_lr_scheduler = None
    
critic_q = TwinQ(obs_shape, action_shape, 256, 2).to(args.device)
critic_v = ValueFunction(obs_shape).to(args.device)
critic_q_optim = torch.optim.Adam(critic_q.parameters(), lr=args.critic_q_lr)
critic_v_optim = torch.optim.Adam(critic_v.parameters(), lr=args.critic_v_lr)

policy = IQLPolicy(
    actor=actor, critic_q=critic_q, critic_v=critic_v, 
    actor_optim=actor_optim, critic_q_optim=critic_q_optim, critic_v_optim=critic_v_optim, 
    expectile=args.expectile, temperature=args.temperature, 
    tau=args.tau, 
    discount=args.discount, 
    max_action=args.max_action, 
    device=args.device, 
).to(args.device)


# main loop
policy.train()
for i_epoch in trange(1, args.max_epoch+1):
    for i_step in range(args.step_per_epoch):
        batch = dataset.sample(args.batch_size)
        train_metrics = policy.update(batch)
        if actor_lr_scheduler is not None:
            actor_lr_scheduler.step()
    
    if i_epoch % args.eval_interval == 0:
        eval_metrics = eval_policy(env, policy, args.eval_episode, seed=args.seed)
    
        logger.info(f"Epicode {i_epoch}: \n{eval_metrics}")

    if i_epoch % args.log_interval == 0:
        logger.log_scalars("", train_metrics, step=i_epoch)
        logger.log_scalars("Eval", eval_metrics, step=i_epoch)

    if i_epoch % args.save_interval == 0:
        logger.log_object(name=f"policy_{i_epoch}.pt", object=policy.state_dict(), path=f"./out/iql/{args.task}/{args.name}_seed{args.seed}/policy/")
    
        