from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch
from omni.isaac.lab.envs.common import VecEnvObs
from omni.isaac.lab.envs.utils.spaces import spec_to_gym_space
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

        # store inputs to class
        self.cfg = cfg

        self.seed(42)
        self.sim = self.world

        # print useful information
        print("[INFO]: Base environment:")
        print(f"\tEnvironment device    : {self.device}")
        print(f"\tEnvironment seed      : {self.cfg.seed}")
        print(f"\tPhysics step-size     : {self.physics_dt}")
        print(f"\tRendering step-size   : {self.physics_dt * self.cfg.sim.render_interval}")
        print(f"\tEnvironment step-size : {self.step_dt}")

        assert self.cfg.sim.render_interval >= self.cfg.decimation, "Render interval should not be smaller than decimation, this will cause multiple render calls."

        # -- init buffers
        self.reset_terminated = False
        self.reset_time_outs = False
        self.reset_buf = False

        # setup the action and observation spaces for Gym
        self.observation_space = spec_to_gym_space(self.cfg.observation_space)
        self.action_space = spec_to_gym_space(self.cfg.action_space)

        # print the environment information
        print("[INFO]: Completed setting up the environment...")
        self.ant_controller = RlAntController(parent_env=self, env_cfg=self.cfg)

        self.obs_buf = None
        self.reward_buf = None

    def __del__(self):
        """Cleanup for the environment."""
        self.close()

    """
    Properties.
    """

    @property
    def num_envs(self) -> int:
        """The number of instances of the environment that are running."""
        return 1

    @property
    def physics_dt(self) -> float:
        """The physics time-step (in s).

        This is the lowest time-decimation at which the simulation is happening.
        """
        return self.cfg.sim.dt

    @property
    def step_dt(self) -> float:
        """The environment stepping time-step (in s).

        This is the time-step at which the environment steps forward.
        """
        return self.cfg.sim.dt * self.cfg.decimation

    @property
    def device(self):
        """The device on which the environment is running."""
        return self.sim.device


    """
    Operations.
    """

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> VecEnvObs:

        super().reset(seed, options)

        # reset state of scene
        self.ant_controller.reset_idx()

        # update articulation kinematics
        self.ant_controller.robot.write_data_to_sim()

        # return observations
        return self.ant_controller.get_observations()

    def pre_step(self, action):
        action = action.to(self.device)

        # process actions
        self.ant_controller.pre_physics_step(action)

    def pre_sub_step(self):
        # set actions into buffers
        self.ant_controller.apply_action()
        # set actions into simulator
        self.ant_controller.robot.write_data_to_sim()

    def sub_step(self):

        # simulate
        self.sim.step(render=False)
        # render between steps only if the GUI or an RTX sensor needs it
        # note: we assume the render interval to be the shortest accepted rendering interval.
        #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
        # if self._sim_step_counter % round(self.cfg.sim.render_interval) == 0:

    def post_sub_step(self):

        # update buffers at sim dt
        self.ant_controller.robot.update(self.physics_dt)

    def post_step(self):
        self.sim.render()

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.ant_controller.episode_length += 1  # step in current episode

        self.reset_terminated, self.reset_time_outs = self.ant_controller.get_dones()
        self.reset_buf = self.reset_terminated or self.reset_time_outs
        self.reward_buf = self.ant_controller.get_rewards(self.reset_terminated)

        # -- reset env if terminated/timed-out and log the episode information
        if self.reset_buf:
            self.ant_controller.reset_idx()
            # update articulation kinematics
            self.ant_controller.robot.write_data_to_sim()

        # update observations
        self.obs_buf = self.ant_controller.get_observations()

        # return observations, rewards, resets and extras
        return self.obs_buf

    def step(self, action: torch.Tensor):

        self.pre_step(action)

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self.pre_sub_step()
            self.sub_step()
            self.post_sub_step()

        return self.post_step()

    def close(self):
        # clear callbacks and instance
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()

    def post_init(self):
        self.ant_controller.post_init()
