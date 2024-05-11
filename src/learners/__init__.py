from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .actor_critic_learner import ActorCriticLearner
from .actor_critic_pac_learner import PACActorCriticLearner
from .actor_critic_pac_dcg_learner import PACDCGLearner
from .maddpg_learner import MADDPGLearner
from .maser_q_learner import MASERQLearner
from .ppo_learner import PPOLearner
from .happo_learner import HAPPOLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner

REGISTRY = {"q_learner": QLearner,
            "coma_learner": COMALearner,
            "qtran_learner": QTranLearner,
            "actor_critic_learner": ActorCriticLearner,
            "maddpg_learner": MADDPGLearner,
            "maser_q_learner": MASERQLearner,
            "ppo_learner": PPOLearner,
            "happo_learner": HAPPOLearner,
            "pac_learner": PACActorCriticLearner,
            "pac_dcg_learner": PACDCGLearner,
            "dmaq_qatten_learner": DMAQ_qattenLearner
            }
