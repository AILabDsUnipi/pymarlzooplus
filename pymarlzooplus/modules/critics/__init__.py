from .coma import COMACritic
from .centralV import CentralVCritic
from .coma_ns import COMACriticNS
from .centralV_ns import CentralVCriticNS
from .maddpg import MADDPGCritic
from .maddpg_ns import MADDPGCriticNS
from .ac import ACCritic
from .ac_ns import ACCriticNS
from .mat import MATCritic
from .happo import HAPPOCritic
from .commformer_critic import CommFormerCritic

REGISTRY = {"coma_critic": COMACritic,
            "cv_critic": CentralVCritic,
            "coma_critic_ns": COMACriticNS,
            "cv_critic_ns": CentralVCriticNS,
            "maddpg_critic": MADDPGCritic,
            "maddpg_critic_ns": MADDPGCriticNS,
            "ac_critic": ACCritic,
            "ac_critic_ns": ACCriticNS,
            "mat_critic": MATCritic,
            "happo_critic": HAPPOCritic,
            "commformer_critic": CommFormerCritic
            }
