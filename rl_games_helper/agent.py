import gym.spaces  # needed for rl-games incompatibility: https://github.com/Denys88/rl_games/issues/261
import omni.isaac.lab_tasks  # noqa: F401
from rl_games.common import env_configurations
from rl_games.common.player import BasePlayer
from rl_games.common.vecenv import IVecEnv
from rl_games.torch_runner import Runner

from rl_games_helper.locomotion_env import LocomotionEnv


def register_rl_games_env(observation_space, action_space):
    class RlGamesVecEnvWrapper(IVecEnv):

        def __init__(self, ):
            # store provided arguments
            self._rl_device = "cpu"
            self._clip_obs = float('inf')
            self._clip_actions = 1.0
            self._sim_device = "cpu"

            self.observation_space = gym.spaces.Box(-self._clip_obs, self._clip_obs, observation_space.shape)
            env_configurations.register("rlgpu",
                                        {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: self})

            self.action_space = gym.spaces.Box(-self._clip_actions, self._clip_actions, action_space.shape)


    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper",
                                              "env_creator": lambda **kwargs: RlGamesVecEnvWrapper()})


class Agent:
    def __init__(self, env: LocomotionEnv, agent_cfg, resume_path, num_envs=1):
        # wrap around environment for rl-games
        register_rl_games_env(env.observation_space, env.action_space)
        # load previously trained model
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

        # set number of actors into agent config
        agent_cfg["params"]["config"]["num_actors"] = num_envs

        # create runner from rl-games
        runner = Runner()
        runner.load(agent_cfg)
        # obtain the agent from the runner
        agent: BasePlayer = runner.create_player()
        agent.restore(resume_path)

        self.agent = agent
        agent.reset()

    def init(self, obs):

        # required: enables the flag for batched observations
        _ = self.agent.get_batch_size(obs, 1)

    def get_action(self, obs):
        # convert obs to agent format
        obs = self.agent.obs_to_torch(obs)
        # agent stepping
        return self.agent.get_action(obs, is_deterministic=self.agent.is_deterministic)

