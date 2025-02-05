from sim.app import init_app, get_app

init_app(headless=False)
app = get_app()

from animals.ant.pretrained_rl_ant_controller import PretrainedRlAntController
from sim.isaac_env import IsaacEnv

from utils import add_gamepad_callback, set_active_camera
from torch import Tensor
from ant_env_cfg import AntEnvCfg
import omni.isaac.lab_tasks  # noqa: F401
from drone.manipulators import ManipulatorState

import omni.isaac.lab_tasks  # noqa: F401

"""Play with RL-Games agent."""


class AntEnv(IsaacEnv):
    def __init__(self, cfg: AntEnvCfg):
        super().__init__(layout_type="grid")

        self.sim = self.world

        # print the environment information
        print("[INFO]: Completed setting up the environment...")
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


# create drone controller
# drone_controller = DroneControllerQGroundControl(env)


def sha1_array(arr: Tensor):
    import hashlib

    # Ensure tensor is in CPU memory and convert to NumPy
    from numpy import ndarray
    tensor_np: ndarray = arr.detach().cpu().numpy()

    # Convert tensor to bytes
    tensor_bytes = tensor_np.tobytes()

    # Compute SHA-1 hash
    sha1_hash = hashlib.sha1(tensor_bytes).hexdigest()

    return sha1_hash


set_active_camera("/World/drone/arm/arm_base/Camera")

mani_state = ManipulatorState()
add_gamepad_callback(mani_state.gamepad_callback)
# reset environment
env.reset()

# simulate environment
# note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
#   attempt to have complete control over environment stepping. However, this removes other
#   operations such as masking that is used for multi-agent learning by RL-Games.
count = 0

while app.is_running():

    # drone_controller.step(mani_state.as_action())
    count += 1
    if count == 4:
        pass
        # assert sha1_array(env.ant_controller.last_obs) == "3cbeb8f5a1e73b228b90ffdbca2a073b0557bedd"
    for _ in range(4):
        env.step(None)

    env.sim.render()
    obs = env.ant_controller.last_obs

last_sha = sha1_array(env.ant_controller.last_obs)
# close the simulator
env.close()

app.close()
