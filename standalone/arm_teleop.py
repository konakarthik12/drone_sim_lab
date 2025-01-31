from sim.app import init_app

init_app()

from utils import add_gamepad_callback

from typing import Any

from gymnasium.core import ObsType

from sim.isaac_env import IsaacEnv

from drone.drone_controller_fixed import DroneControllerFixed
from omni.isaac.lab.assets import Articulation

from animals.ant.ant_usd_cfg import get_ant_cfg


class ArmTeleOpEnv(IsaacEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.drone_controller = DroneControllerFixed(parent_env=self)
        cfg = get_ant_cfg(init_pos=(0.01569,0.00481,0.1),scale=[0.045] * 3)
        self.object = Articulation(cfg)

    # actual left and right joystick values are handled by the QGroundControl
    # we only handle manipulator control here
    # action space (3,): [joint1_pos, joint2_pos, gripper_close]
    def step(self, action):
        super().step(action)
        print(action)
        self.drone_controller.step(action)

        # self.crab_controller.step()
        return {}, 0.0, False, False, {}

    def post_init(self):
        self.drone_controller.post_init()

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.drone_controller.reset()
        # self.crab_controller.reset()
        return {}, {}


env = ArmTeleOpEnv()

from drone.manipulators import ManipulatorState

state = ManipulatorState(joint2_pos=-1.5)
add_gamepad_callback(state.gamepad_callback)


def run_simulation():
    env.reset()

    while env.app.is_running():
        env.step(state.as_action())


max_steps = 10000

run_simulation()
env.close()
