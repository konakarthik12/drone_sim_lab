from animals.ant.ant_env_cfg import AntEnvCfg
from animals.ant.pretrained_rl_ant_controller import PretrainedRlAntController
from sim.isaac_env import IsaacEnv

class AntEnv(IsaacEnv):
    def __init__(self, cfg: AntEnvCfg):
        super().__init__(layout_type="grid")

        self.sim = self.world
        self.ant_controller = PretrainedRlAntController(parent_env=self, env_cfg=cfg)

    def reset(self, seed=None, options=None):
        super().reset(seed, options)
        self.ant_controller.reset()

    def step(self, _):
        self.ant_controller.pre_step()
        super().step(None)
        self.ant_controller.post_step()

    def post_init(self):
        self.ant_controller.post_init()

