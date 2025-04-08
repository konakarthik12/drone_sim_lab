from omni.isaac.lab_tasks.direct.ant.ant_env import AntEnvCfg

from animals.ant.pretrained_rl_ant_controller import PretrainedRlAgentController
from sim.isaac_env import IsaacEnv


class AgentEnv(IsaacEnv):
    def __init__(self, cfg: AntEnvCfg):
        super().__init__(layout="grid")

        self.sim = self.world
        self.animal_controller = PretrainedRlAgentController(parent_env=self, cfg=cfg)

    def reset(self, seed=None, options=None):
        super().reset(seed, options)
        self.animal_controller.reset()

    def step(self, _):
        self.animal_controller.pre_step()
        super().step(None)
        self.animal_controller.post_step()

    def post_init(self):
        self.animal_controller.post_init()
