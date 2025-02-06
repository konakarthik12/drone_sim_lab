from sim.app import init_app, get_app

init_app(headless=False)
app = get_app()

from utils import add_gamepad_callback, set_active_camera
from animals.ant.ant_env_cfg import AntEnvCfg
from drone.manipulators import ManipulatorState
from animals.ant.ant_env import AntEnv

env_cfg = AntEnvCfg()
env = AntEnv(env_cfg)

from drone.drone_controller_qgroundcontrol import DroneControllerQGroundControl

# create drone controller
drone_controller = DroneControllerQGroundControl(env)

mani_state = ManipulatorState()
add_gamepad_callback(mani_state.gamepad_callback)

set_active_camera("/World/drone/arm/arm_base/Camera")
env.reset()
drone_controller.post_init()
while app.is_running():

    drone_controller.step(mani_state.as_action())
    for _ in range(4):
        env.step(None)

    env.sim.render()
    obs = env.ant_controller.last_obs

env.close()
app.close()
