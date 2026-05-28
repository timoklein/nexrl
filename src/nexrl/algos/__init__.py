"""RL algorithm implementations.

On-policy: ppo, ppg, ppo_icm, ppo_e3b, ppo_rnd, ppo_rnn, vmpo, vmpo_continuous
Off-policy continuous: sac, sac_simba, td3, mpo_continuous
Off-policy discrete: dqn, iqn, pqn, rainbow, mpo
Model-based: alphazero, muzero, dreamer, sgf
Off-policy dual-mode: mrq, mrsq
Off-policy visual: drqv2
"""

from .alphazero import train_alphazero
from .bbf import train_bbf
from .dqn import train_dqn
from .dreamer import train_dreamer
from .drqv2 import train_drqv2
from .iqn import train_iqn
from .mpo import train_mpo
from .mpo_continuous import train_mpo_continuous
from .mrq import train_mrq
from .mrsq import train_mrsq
from .muzero import train_muzero
from .ppg import train_ppg
from .ppo import train_ppo
from .ppo_e3b import train_ppo_e3b
from .ppo_icm import train_ppo_icm
from .ppo_rnd import train_ppo_rnd
from .ppo_rnn import train_ppo_rnn
from .pqn import train_pqn
from .rainbow import train_rainbow
from .sac import train_sac
from .sac_simba import train_sac_simba
from .sgf import train_sgf
from .td3 import train_td3
from .vmpo import train_vmpo
from .vmpo_continuous import train_vmpo_continuous


__all__ = [
    "train_alphazero",
    "train_bbf",
    "train_dqn",
    "train_dreamer",
    "train_drqv2",
    "train_iqn",
    "train_mpo",
    "train_mpo_continuous",
    "train_mrq",
    "train_mrsq",
    "train_muzero",
    "train_ppg",
    "train_ppo",
    "train_ppo_e3b",
    "train_ppo_icm",
    "train_ppo_rnd",
    "train_ppo_rnn",
    "train_pqn",
    "train_rainbow",
    "train_sac",
    "train_sac_simba",
    "train_sgf",
    "train_td3",
    "train_vmpo",
    "train_vmpo_continuous",
]
