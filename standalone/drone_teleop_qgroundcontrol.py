from sim.app import init_app

init_app()

from carb.input import GamepadInput
from drone.teleop.drone_teleop_env import DroneTeleOpEnv
from utils import add_gamepad_callback, set_active_camera, GamepadButtonPressWatcher

env = DroneTeleOpEnv(layout_type="grid", record_path="drone_commands.pkl")

from drone.manipulators import ManipulatorState

state = ManipulatorState()
add_gamepad_callback(state.gamepad_callback)

exit_watcher = GamepadButtonPressWatcher(GamepadInput.MENU1)
add_gamepad_callback(exit_watcher.gamepad_callback)

commands = []

set_active_camera("/World/drone/arm/arm_base/Camera")
env.reset()
while not exit_watcher.pressed:
    env.step(state.as_action())
env.close()
