from typing import Any

from gymnasium.core import ObsType

from animals.ant.ant_controller import AntController
from animals.ant.ant_usd_cfg import get_ant_cfg
from animals.ant.pretrained_rl_ant_controller import PretrainedRlAntController
from drone.drone_controller_qgroundcontrol import DroneControllerQGroundControl
from sim.isaac_env import IsaacEnv


class DroneTeleOpEnv(IsaacEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.drone_controller = DroneControllerQGroundControl(parent_env=self)
        self.ant_controller = AntController(parent_env=self, ant_cfg=get_ant_cfg((6.0, 0.0, 0.5*0.045), [0.045] * 3))

    # actual left and right joystick values are handled by the QGroundControl
    # we only handle manipulator control here
    # action space (3,): [joint1_pos, joint2_pos, gripper_close]
    def step(self, action):
        self.ant_controller.pre_step()
        super().step(action)
        self.drone_controller.step(action)
        self.ant_controller.post_step()
        return {}, 0.0, False, False, {}

    def post_init(self):
        self.drone_controller.post_init()
        self.ant_controller.post_init()

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.drone_controller.reset()
        self.ant_controller.reset()
        return {}, {}

    def close(self):
        self.drone_controller.close()
