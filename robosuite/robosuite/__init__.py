import os

from robosuite.environments.base import make
from robosuite.environments.sawyer_lift import SawyerLift
from robosuite.environments.sawyer_stack import SawyerStack
from robosuite.environments.sawyer_pick_place import SawyerPickPlace
from robosuite.environments.sawyer_bin_picking import SawyerBinPicking
from robosuite.environments.sawyer_bin_picking_gqcnn import SawyerBinPickingGQCNN
from robosuite.environments.sawyer_nut_assembly import SawyerNutAssembly

from robosuite.environments.baxter_lift import BaxterLift
from robosuite.environments.baxter_peg_in_hole import BaxterPegInHole
from robosuite.environments.baxter_bin_picking import BaxterBinPicking
from robosuite.environments.baxter_bin_picking_gqcnn import BaxterBinPickingGQCNN
from robosuite.environments.baxter_bin_picking_concave import BaxterBinPickingConcave
from robosuite.environments.baxter_collect_data import BaxterCollectData
from robosuite.environments.baxter_steeped_bin_collect_data import BaxterSteepedBinCollectData
from robosuite.environments.baxter_active_sensing import BaxterActiveSensing

from robosuite.environments.swimmer_collect_data import SwimmerCollectData

from robosuite.environments.baxter_push_rl import BaxterPush

__version__ = "0.1.0"
__logo__ = """
      ;     /        ,--.
     ["]   ["]  ,<  |__**|
    /[_]\  [~]\/    |//  |
     ] [   OOO      /o|__|
"""
