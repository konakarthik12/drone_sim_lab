import numpy as np

from sim.app import init_app

init_app()

from carb.input import GamepadInput
from drone.teleop.drone_teleop_env import DroneTeleOpEnv
from utils import add_gamepad_callback, set_active_camera, GamepadButtonPressWatcher, save_pickle

env = DroneTeleOpEnv(layout_type="grid")

from drone.manipulators import ManipulatorState

state = ManipulatorState()
add_gamepad_callback(state.gamepad_callback)

exit_watcher = GamepadButtonPressWatcher(GamepadInput.MENU1)
add_gamepad_callback(exit_watcher.gamepad_callback)

commands = []

set_active_camera("/World/drone/arm/arm_base/Camera")
env.reset()
count = 0
while not exit_watcher.pressed:
    mani_action = state.as_action()
    env.step(mani_action)
    if count % 4 == 0:
        env.sim.render()
    drone_action = np.array(env.drone_controller.backend.input_reference())
    commands.append(np.append(drone_action, mani_action))

save_pickle("episode_actions.pkl", commands)
print("Commands saved to episode_actions.pkl")
env.close()
