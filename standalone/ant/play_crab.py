from animals.pretrained_rl_agent_controller import PretrainedRlAgentController
from sim.app import init_app, get_app

init_app(headless=False)
app = get_app()

from utils import add_gamepad_callback, set_active_camera
from omni.isaac.lab_tasks.direct.crab.crab_env import CrabEnvCfg
from animals.crab.crab_usd_cfg import get_crab_cfg
from drone.manipulators import ManipulatorState
from sim.isaac_env import IsaacEnv


def default_rl_env():
    cfg = CrabEnvCfg()
    cfg.robot = get_crab_cfg(init_pos=(6.0, 0.0, None))  # Keep the z-position same as training environment

    return cfg


class CrabEnv(IsaacEnv):
    def __init__(self, cfg: CrabEnvCfg):
        super().__init__(layout="water")

        self.sim = self.world
        self.controller = PretrainedRlAgentController(parent_env=self, env_cfg=default_rl_env())

    def reset(self, seed=None, options=None):
        super().reset(seed, options)
        self.controller.reset()

    def step(self, _):
        self.controller.pre_step()
        super().step(None)
        self.controller.post_step()

    def post_init(self):
        self.controller.post_init()


env_cfg = CrabEnvCfg()
env = CrabEnv(env_cfg)

# create drone controller
# drone_controller = DroneControllerQGroundControl(env)

mani_state = ManipulatorState()
add_gamepad_callback(mani_state.gamepad_callback)

set_active_camera("/World/drone/arm/arm_base/Camera")
env.reset()
# drone_controller.post_init()
while app.is_running():

    # drone_controller.post_step(mani_state.as_action())
    for _ in range(4):
        env.step(None)

    env.sim.render()
    # obs = env.crab_controller.last_obs

env.close()
app.close()
