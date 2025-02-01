from sim.app import init_app
init_app()

from drone.teleop.drone_teleop_env import DroneTeleOpEnv
from utils import add_gamepad_callback

env = DroneTeleOpEnv(layout_type="grid")

from drone.manipulators import ManipulatorState

state = ManipulatorState()
add_gamepad_callback(state.gamepad_callback)


def run_simulation():
    env.reset()

    while env.app.is_running():
        env.step(state.as_action())


max_steps = 10000

run_simulation()
env.close()
