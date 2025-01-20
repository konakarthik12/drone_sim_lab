from typing import Any

from gymnasium.core import ObsType

from animals.ant.zero_ant_controller import ZeroAntController
from drone.drone_controller_qgroundcontrol import DroneControllerQGroundControl
from sim.isaac_env import IsaacEnv


class DroneTeleOpEnv(IsaacEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ant_controller = ZeroAntController(parent_env=self)
        self.drone_controller = DroneControllerQGroundControl(parent_env=self)

    # actual left and right joystick values are handled by the QGroundControl
    # we only handle manipulator control here
    # action space (3,): [joint1_pos, joint2_pos, gripper_close]
    def step(self, action):
        super().step(action)
        self.drone_controller.step(action)
        # self.crab_controller.step()
        return {}, 0.0, False, False, {}

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
