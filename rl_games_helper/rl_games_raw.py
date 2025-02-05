# needed to import for allowing type-hinting:gym.spaces.Box | None
from __future__ import annotations

import gym.spaces  # needed for rl-games incompatibility: https://github.com/Denys88/rl_games/issues/261
import torch
from omni.isaac.lab.envs import ManagerBasedRLEnv
from rl_games.common import env_configurations
from rl_games.common import vecenv
from rl_games.common.vecenv import IVecEnv

from rl_games_helper.locomotion_env import LocomotionEnv


def register_rl_games_env(env: LocomotionEnv):
    class RlGamesVecEnvWrapper(IVecEnv):

        def __init__(self, ):
            # store provided arguments
            self._rl_device = "cpu"
            self._clip_obs = float('inf')
            self._clip_actions = 1.0
            self._sim_device = "cpu"

            self.observation_space = gym.spaces.Box(-self._clip_obs, self._clip_obs, env.observation_space.shape)
            env_configurations.register("rlgpu",
                                        {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: self})

            self.action_space = gym.spaces.Box(-self._clip_actions, self._clip_actions, env.action_space.shape)


    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                              "env_creator": lambda **kwargs: RlGamesVecEnvWrapper()})
