from reproduce.xql.config.offline.original.base import *

actor_lr = 3e-4
critic_q_lr = 3e-4
critic_v_lr = 3e-4
hidden_dims = [256, 256]
discount = 0.99
tau = 0.005


aw_temperature = 3.0
norm_layer = True
dropout = None
value_dropout = None

actor_opt_decay_schedule = "cosine"