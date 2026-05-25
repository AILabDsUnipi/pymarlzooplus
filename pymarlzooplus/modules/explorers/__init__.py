from pymarlzooplus.modules.explorers.EOI import Explorer as EOIExplorer
from pymarlzooplus.modules.explorers.MAVEN import EZExplorer as MAVENExplorer
from pymarlzooplus.modules.explorers.state_pred_cvae  import StatePredBL, StatePredCVAE

REGISTRY = {
    "eoi": EOIExplorer,
    "maven": MAVENExplorer,
    "statepredbl": StatePredBL,
    "statepredcvae": StatePredCVAE
}
