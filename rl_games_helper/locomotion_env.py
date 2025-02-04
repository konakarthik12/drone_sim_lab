from __future__ import annotations

import math
from typing import Any, ClassVar

import gymnasium as gym
import omni.isaac.core.utils.torch as torch_utils
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

    def __init__(self, cfg: AntEnvCfg, **kwargs):

        super().__init__(layout_type="air")

        # store inputs to class
        self.cfg = cfg

        # initialize internal variables
        self._is_closed = False

        self.seed(42)
        self.cfg.sim.render_interval = 250 / 60
        assert self.cfg.sim.dt == 1 / 250
        self.sim = self.world

        # print useful information
        print("[INFO]: Base environment:")
        print(f"\tEnvironment device    : {self.device}")
        print(f"\tEnvironment seed      : {self.cfg.seed}")
        print(f"\tPhysics step-size     : {self.physics_dt}")
        print(f"\tRendering step-size   : {self.physics_dt * self.cfg.sim.render_interval}")
        print(f"\tEnvironment step-size : {self.step_dt}")

        assert self.cfg.sim.render_interval >= self.cfg.decimation, "Render interval should not be smaller than decimation, this will cause multiple render calls."

        # initialize data and constants
        # -- counter for simulation steps
        self._sim_step_counter = 0
        # -- counter for curriculum
        self.common_step_counter = 0
        # -- init buffers
        self.episode_length = 0
        self.reset_terminated = False
        self.reset_time_outs = False
        self.reset_buf = False

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # print the environment information
        print("[INFO]: Completed setting up the environment...")
        self.ant_controller = RlAntController(parent_env=self, env_cfg=self.cfg)

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

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self):
        """The maximum episode length in steps adjusted from s."""
        return math.ceil(self.max_episode_length_s / (self.cfg.sim.dt * self.cfg.decimation))

    """
    Operations.
    """

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[VecEnvObs, dict]:

        super().reset(seed, options)
        # set the seed
        if seed is not None:
            self.seed(seed)

        # reset state of scene
        self.reset_idx()

        # update articulation kinematics
        self.write_scene_data_to_sim()

        # return observations
        return self._get_observations(), {}

    def pre_step(self, action):
        action = action.to(self.device)

        # process actions
        self._pre_physics_step(action)
    def pre_sub_step(self):
        self._sim_step_counter += 1
        # set actions into buffers
        self._apply_action()
        # set actions into simulator
        self.write_scene_data_to_sim()

    def sub_step(self):

        # simulate
        self.sim.step(render=False)
        # render between steps only if the GUI or an RTX sensor needs it
        # note: we assume the render interval to be the shortest accepted rendering interval.
        #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
        # if self._sim_step_counter % round(self.cfg.sim.render_interval) == 0:
    def post_sub_step(self):

        # update buffers at sim dt
        self.update_scene(dt=self.physics_dt)

    def post_step(self):
        self.sim.render()

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length += 1  # step in current episode
        self.common_step_counter += 1  # total step

        self.reset_terminated, self.reset_time_outs = self._get_dones()
        self.reset_buf = self.reset_terminated or self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # -- reset env if terminated/timed-out and log the episode information
        if self.reset_buf:
            self.reset_idx()
            # update articulation kinematics
            self.write_scene_data_to_sim()

        # update observations
        self.obs_buf = self._get_observations()

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

    @staticmethod
    def seed(seed: int = -1) -> int:
        """Set the seed for the environment.

        Args:
            seed: The seed for random generator. Defaults to -1.

        Returns:
            The seed used for random generator.
        """
        # set seed for replicator
        try:
            import omni.replicator.core as rep

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        # set seed for torch and other libraries
        return torch_utils.set_seed(seed)

    def close(self):
        # clear callbacks and instance
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()


    """
    Helper functions.
    """

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""

        # set up spaces
        self.single_observation_space = gym.spaces.Dict()
        self.single_observation_space["policy"] = spec_to_gym_space(self.cfg.observation_space)
        self.single_action_space = spec_to_gym_space(self.cfg.action_space)

        # batch the spaces for vectorized environments
        self.observation_space = self.single_observation_space["policy"]
        self.action_space = self.single_action_space






    def post_init(self):
        self.ant_controller.post_init()

    def _pre_physics_step(self, actions: torch.Tensor):
        self.ant_controller.pre_physics_step(actions)

    def _apply_action(self):
        self.ant_controller.apply_action()

    def _get_observations(self) -> dict:
        return self.ant_controller.get_observations()

    def _get_rewards(self) -> torch.Tensor:
        return self.ant_controller.get_rewards(self.reset_terminated)

    def _get_dones(self) -> tuple[bool, bool]:
        return self.ant_controller.get_dones(self.episode_length, self.max_episode_length)

    def reset_idx(self):
        self.reset_scene()
        self.episode_length = 0
        self.ant_controller.reset_idx()

    def write_scene_data_to_sim(self):
        self.ant_controller.robot.write_data_to_sim()

    def reset_scene(self):
        self.ant_controller.robot.reset()

    def update_scene(self, dt: float):
        self.ant_controller.robot.update(dt)
