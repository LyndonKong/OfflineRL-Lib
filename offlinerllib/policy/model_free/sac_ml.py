import torch
import torch.optim as optim
from operator import itemgetter

from typing import Union, Tuple, Dict

from offlinerllib.policy.model_free.sacn import SACNPolicy
from offlinerllib.module.actor import BaseActor
from offlinerllib.module.critic import Critic
from offlinerllib.utils.functional import clip_log_pi


class SAC_MLPolicy(SACNPolicy):
    """
    SAC plus Maximum Likelihood
    """
    def __init__(self,
                 actor: BaseActor,
                 critic: Critic,
                 actor_optim: optim.Optimizer,
                 critic_optim: optim.Optimizer,
                 tau: float = 0.005,
                 omega: float = 1.0,
                 gamma: float = 0.99,
                 alpha: Union[float, Tuple[float, torch.Tensor,
                                           optim.Optimizer]] = 0.2,
                 do_reverse_update: bool = False,
                 device: Union[str, torch.device] = "cpu") -> None:
        super().__init__(actor, critic, actor_optim, critic_optim, tau, gamma,
                         alpha, do_reverse_update, device)
        self.omega = omega

    def _actor_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        actor_loss, actor_item_dict = super()._actor_loss(batch)
        obss, actions, next_obss, rewards, terminals = itemgetter("observations", "actions", "next_observations", "rewards", "terminals")(batch)
        raw_log_pi, info = self.actor.evaluate(obss, actions)
        dist = info["dist"]
        log_pi = clip_log_pi(dist, raw_log_pi)
        with torch.no_grad():
            q_mean = self._critic(obss, actions)
            lam = self.onmega/(q_mean.abs().mean())
        actor_loss = lam * actor_loss - log_pi.mean()
        actor_item_dict["info/log_pi"] = log_pi.item()
        return actor_loss, actor_item_dict