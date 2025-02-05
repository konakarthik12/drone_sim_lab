from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch
from omni.isaac.lab.envs.common import VecEnvObs
from omni.isaac.version import get_version

from animals.ant.rl_ant_controller import RlAntController
from rl_games_helper.ant_env_cfg import AntEnvCfg
from sim.isaac_env import IsaacEnv


class LocomotionEnv(IsaacEnv, gym.Env):
    """Whether the environment is a vectorized environment."""
    metadata: dict[str, Any] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    """Metadata for the environment."""

    def __init__(self, cfg: AntEnvCfg):
        super().__init__(layout_type="grid")


        self.sim = self.world


        # print the environment information
        print("[INFO]: Completed setting up the environment...")
        self.ant_controller = RlAntController(parent_env=self, env_cfg=cfg)


    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> VecEnvObs:
        super().reset(seed, options)
        self.ant_controller.last_obs = self.ant_controller.reset()
        self.ant_controller.agent.init(self.ant_controller.last_obs)



    def step(self, _):

        self.ant_controller.pre_step()
        # simulate
        super().step(None)
        self.ant_controller.post_step()

    def post_init(self):
        self.ant_controller.post_init()
