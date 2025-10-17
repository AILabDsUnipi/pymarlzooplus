from pymarlzooplus.modules.explorers.EOI import Explorer as EOIExplorer
from pymarlzooplus.modules.explorers.MAVEN import EZExplorer as MAVENExplorer

REGISTRY = {
    "eoi": EOIExplorer,
    "maven": MAVENExplorer
}
