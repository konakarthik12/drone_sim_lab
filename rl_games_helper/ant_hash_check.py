from sim.app import init_app, get_app

init_app(headless=False)
app = get_app()

from animals.ant.pretrained_rl_ant_controller import PretrainedRlAntController
from sim.isaac_env import IsaacEnv

from torch import Tensor
from ant_env_cfg import AntEnvCfg
import omni.isaac.lab_tasks  # noqa: F401

import omni.isaac.lab_tasks  # noqa: F401

"""Play with RL-Games agent."""


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


env_cfg = AntEnvCfg()
env = AntEnv(env_cfg)


def sha1_array(arr: Tensor):
    import hashlib

    from numpy import ndarray
    tensor_np: ndarray = arr.detach().cpu().numpy()
    tensor_bytes = tensor_np.tobytes()
    sha1_hash = hashlib.sha1(tensor_bytes).hexdigest()

    return sha1_hash


env.reset()

for _ in range(3):
    # decimation = 4
    for _ in range(4):
        env.step(None)
assert sha1_array(env.ant_controller.last_obs) == "3cbeb8f5a1e73b228b90ffdbca2a073b0557bedd"

env.close()
app.close()
