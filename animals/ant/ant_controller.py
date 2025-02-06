from omni.isaac.lab.assets import ArticulationCfg

from animals.ant.ant_usd_cfg import get_ant_cfg
from sim.controller import Controller


class AntController(Controller):
    def __init__(self, parent_env, ant_cfg: ArticulationCfg):
        from omni.isaac.lab.assets import Articulation

        self.world = parent_env.world
        self.robot = Articulation(ant_cfg)
