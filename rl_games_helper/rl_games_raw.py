
# needed to import for allowing type-hinting:gym.spaces.Box | None
from __future__ import annotations

import gym.spaces  # needed for rl-games incompatibility: https://github.com/Denys88/rl_games/issues/261
import torch
from omni.isaac.lab.envs import ManagerBasedRLEnv
from rl_games.common import env_configurations
from rl_games.common import vecenv
from rl_games.common.vecenv import IVecEnv

from direct_rl_env import DirectRLEnv


class RlGamesVecEnvWrapper(IVecEnv):


    def __init__(self, env: DirectRLEnv):

        assert isinstance(env.unwrapped, DirectRLEnv), "The environment must be inherited from DirectRLEnv."
        # initialize the wrapper
        self.env = env
        # store provided arguments
        self._rl_device = "cuda:0"
        self._clip_obs = float('inf')
        self._clip_actions = 1.0
        self._sim_device = env.unwrapped.device

        env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: self})

    @property
    def observation_space(self) -> gym.spaces.Box:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        # note: rl-games only wants single observation space
        policy_obs_space = self.unwrapped.single_observation_space["policy"]

        # note: maybe should check if we are a sub-set of the actual space. don't do it right now since
        #   in ManagerBasedRLEnv we are setting action space as (-inf, inf).
        return gym.spaces.Box(-self._clip_obs, self._clip_obs, policy_obs_space.shape)

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        # note: rl-games only wants single action space
        action_space = self.unwrapped.single_action_space

        # return casted space in gym.spaces.Box (OpenAI Gym)
        # note: maybe should check if we are a sub-set of the actual space. don't do it right now since
        #   in ManagerBasedRLEnv we are setting action space as (-inf, inf).
        return gym.spaces.Box(-self._clip_actions, self._clip_actions, action_space.shape)

    @property
    def unwrapped(self) -> ManagerBasedRLEnv | DirectRLEnv:

        return self.env.unwrapped

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self):  # noqa: D102
        obs_dict, _ = self.env.reset()
        # process observations and states
        return self.post_step(obs_dict)

    def pre_step(self, actions):  # noqa: D102
        # move actions to sim-device
        actions = actions.detach().clone().to(device=self._sim_device)
        # clip the actions
        actions = torch.clamp(actions, -self._clip_actions, self._clip_actions)
        return actions

    def post_step(self, obs_dict):
        # process policy obs
        obs = obs_dict["policy"]
        # clip the observations
        obs = torch.clamp(obs, -self._clip_obs, self._clip_obs)
        # move the buffer to rl-device
        obs = obs.to(device=self._rl_device).clone()

        return obs



    def close(self):  # noqa: D102
        return self.env.close()




"""
Environment Handler.
"""


class RlGamesGpuEnv(IVecEnv):
    """Thin wrapper to create instance of the environment to fit RL-Games runner."""

    def __init__(self, config_name: str, _: int, **kwargs):
        """Initialize the environment.

        Args:
            config_name: The name of the environment configuration.
            num_actors: The number of actors in the environment. This is not used in this wrapper.
        """
        self.env: RlGamesVecEnvWrapper = env_configurations.configurations[config_name]["env_creator"](**kwargs)

    def step(self, action):  # noqa: D102
        return self.env.step(action)

    def reset(self):  # noqa: D102
        return self.env.reset()

    def get_number_of_agents(self) -> int:
        """Get number of agents in the environment.

        Returns:
            The number of agents in the environment.
        """
        return self.env.get_number_of_agents()

    def get_env_info(self) -> dict:
        """Get the Gym spaces for the environment.

        Returns:
            The Gym spaces for the environment.
        """
        return self.env.get_env_info()


# register the environment to rl-games registry
# note: in agents configuration: environment name must be "rlgpu"
vecenv.register(
    "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
)
