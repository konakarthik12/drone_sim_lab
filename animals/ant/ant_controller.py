from omni.isaac.lab.assets import ArticulationCfg

from animals.art_controller import ArtController


class AntController(ArtController):
    def __init__(self, parent_env, ant_cfg: ArticulationCfg):
        super().__init__(parent_env, ant_cfg)
