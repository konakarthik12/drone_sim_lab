from sim.app import init_app
init_app()

import numpy as np

from drone.teleop.drone_teleop_env import DroneTeleOpEnv
from utils import add_gamepad_callback

env = DroneTeleOpEnv(layout_type="grid")


class ManipulatorState:
    def __init__(self):
        self.joint1_pos = 0
        self.joint2_pos = 0
        self.gripper = 0.0

    def gamepad_callback(self, event):
        import carb.input

        if event.input == carb.input.GamepadInput.X:
            if event.value > 0.5:
                self.gripper = 1 - self.gripper
        if event.input == carb.input.GamepadInput.DPAD_UP:
            self.joint1_pos += 0.1
        if event.input == carb.input.GamepadInput.DPAD_DOWN:
            self.joint1_pos -= 0.1
        if event.input == carb.input.GamepadInput.LEFT_SHOULDER:
            self.joint2_pos -= 0.05
        if event.input == carb.input.GamepadInput.RIGHT_SHOULDER:
            self.joint2_pos += 0.05

    def as_action(self):
        return np.array([self.joint1_pos, self.joint2_pos, self.gripper])


state = ManipulatorState()
add_gamepad_callback(state.gamepad_callback)


def run_simulation():
    env.reset()

    while env.app.is_running():
        env.step(state.as_action())


max_steps = 10000

run_simulation()
env.close()
