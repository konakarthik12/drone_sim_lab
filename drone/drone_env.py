from typing import Any

from gymnasium.core import ObsType

from animals.ant.ant_controller import AntController
from animals.ant.ant_usd_cfg import get_ant_cfg
from animals.crab.zero_crab_controller import ZeroCrabController
from drone.drone_controller import DroneController
from drone.drone_controller_qgroundcontrol import DroneControllerQGroundControl
from sim.isaac_env import IsaacEnv


class DroneEnv(IsaacEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ant_controller = AntController(parent_env=self, ant_cfg=get_ant_cfg((6.0, 0.0, 0.0), [0.045] * 3))
        self.drone_controller = DroneController(parent_env=self)

    def step(self, action):
        super().step(action)
        self.drone_controller.step(action)
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
        return {}, {}

    def close(self):
        self.drone_controller.close()
