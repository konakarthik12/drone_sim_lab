import omni.isaac.lab_tasks  # noqa: F401
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from rl_games_helper.locomotion_env import LocomotionEnv
from rl_games_helper.rl_games_raw import register_rl_games_env


class Agent:
    def __init__(self, env: LocomotionEnv, agent_cfg, resume_path, num_envs=1):
        # wrap around environment for rl-games
        register_rl_games_env(env)
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

