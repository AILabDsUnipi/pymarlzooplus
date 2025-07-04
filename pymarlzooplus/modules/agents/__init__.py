from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .mlp_mat_agent import MLPMATAgent
from .rnn_agent_happo import RNNAgentHAPPO
from .rnn_agent_emc import RNNAgentEMC
from .rnn_agent_cds import RNNAgentCDS
from .rnn_maven_agent import RNNAgentMaven

REGISTRY = {"rnn": RNNAgent,
            "rnn_ns": RNNNSAgent,
            "mlp_mat": MLPMATAgent,
            "rnn_happo": RNNAgentHAPPO,
            "rnn_emc": RNNAgentEMC,
            "rnn_cds": RNNAgentCDS,
            "rnn_maven": RNNAgentMaven
            }


