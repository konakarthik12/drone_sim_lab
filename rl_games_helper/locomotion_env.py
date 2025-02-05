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

        self.cfg = cfg

        self.sim = self.world

        assert self.cfg.sim.render_interval >= self.cfg.decimation, "Render interval should not be smaller than decimation, this will cause multiple render calls."

        # print the environment information
        print("[INFO]: Completed setting up the environment...")
        self.ant_controller = RlAntController(parent_env=self, env_cfg=self.cfg)
        self.current_step = 0
        self.last_obs = None

    def __del__(self):
        """Cleanup for the environment."""
        self.close()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> VecEnvObs:
        super().reset(seed, options)
        self.last_obs = self.ant_controller.reset()

    def pre_step(self):
        if self.current_step % self.cfg.decimation == 0:
            action = self.ant_controller.agent.get_action(self.last_obs)
            # process actions
            self.ant_controller.pre_physics_step(action)
            # set actions into buffers
        self.ant_controller.apply_action()

    def post_step(self):
        # render between steps only if the GUI or an RTX sensor needs it
        # note: we assume the render interval to be the shortest accepted rendering interval.
        #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
        # if self._sim_step_counter % round(self.cfg.sim.render_interval) == 0:
        # update buffers at sim dt
        self.ant_controller.update()
        if self.current_step % self.cfg.decimation == 0:
            self.sim.render()
        self.current_step += 1
        if self.current_step % self.cfg.decimation == 0:
            self.last_obs = self.ant_controller.post_step()

    def step(self, _):

        self.pre_step()
        # simulate
        super().step(None)
        self.post_step()


    def close(self):
        # clear callbacks and instance
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()

    def post_init(self):
        self.ant_controller.post_init()
