from omni.isaac.lab.assets import ArticulationCfg

from animals.art_controller import ArtController


class ZeroArtController(ArtController):
    def __init__(self, parent_env, art_cfg: ArticulationCfg):
        super().__init__(parent_env, art_cfg)
        self.joint_pos = None

    def post_init(self):
        self.joint_pos = self.robot.data.joint_effort_target
        assert self.joint_pos is not None

    def pre_step(self, *args, **kwargs):
        self.robot.set_joint_position_target(self.joint_pos)
