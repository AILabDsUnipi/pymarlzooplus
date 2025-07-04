from .q_learner import QLearner
from .coma_learner import COMALearner
from .emc_qplex_curiosity_vdn_learner import EMC_qplex_curiosity_vdn_Learner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .maddpg_learner import MADDPGLearner
from .ppo_learner import PPOLearner
from .happo_learner import HAPPOLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .mat_learner import MATLearner
from .maser_q_learner import  MASERQLearner
from .maven_learner import MavenLearner

REGISTRY = {"q_learner": QLearner,
            "coma_learner": COMALearner,
            "qtran_learner": QTranLearner,
            "actor_critic_learner": ActorCriticLearner,
            "maddpg_learner": MADDPGLearner,
            "ppo_learner": PPOLearner,
            "happo_learner": HAPPOLearner,
            "dmaq_qatten_learner": DMAQ_qattenLearner,
            "mat_learner": MATLearner,
            "emc_qplex_curiosity_vdn_learner": EMC_qplex_curiosity_vdn_Learner,
            "maser_q_learner": MASERQLearner,
            "maven_learner": MavenLearner
            }
