from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC
from .mat_controller import MATMAC
from .happo_controller import happoMAC
from .emc_controller import emcMAC
from .cds_controller import cdsMAC
from .maven_controller import Maven

REGISTRY = {"basic_mac": BasicMAC,
            "non_shared_mac": NonSharedMAC,
            "maddpg_mac": MADDPGMAC,
            "mat_mac": MATMAC,
            "happo_mac": happoMAC,
            "emc_mac": emcMAC,
            "cds_mac": cdsMAC,
            "maven_mac": Maven
            }

