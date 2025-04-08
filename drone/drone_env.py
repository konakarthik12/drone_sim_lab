from typing import Any

from gymnasium.core import ObsType

from animals.pretrained_rl_agent_controller import PretrainedRlAgentController
from drone.drone_controller import DroneController
from sim.isaac_env import IsaacEnv


class DroneEnv(IsaacEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.drone_controller = DroneController(parent_env=self)
        self.animal_controller = PretrainedRlAgentController(parent_env=self)

    def step(self, action):
        self.animal_controller.pre_step()
        super().step(action)
        self.drone_controller.post_step(action)
        self.animal_controller.post_step()
        return {}, 0.0, False, False, {}

    def post_init(self):
        self.drone_controller.post_init()
        self.animal_controller.post_init()

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.drone_controller.reset()
        self.animal_controller.reset()
        return {}, {}

    def close(self):
        self.drone_controller.close()
