from omni.isaac.lab_tasks.utils import load_cfg_from_registry

from animals.ant.ant_env_cfg import AntEnvCfg
from animals.crab.crab_env_cfg import CrabEnvCfg
from animals.ant.rl_ant_controller import RlAntController
from sim.isaac_env import IsaacEnv

# TASK_NAME = "Isaac-Ant-Direct-v0"
# RESUME_PATH = "/home/kkona/Documents/research/drone_sim_lab/assets/animals/ant_direct_policy.pth"
TASK_NAME = "Isaac-Crab-Direct-v0"
RESUME_PATH = "/home/kkona/Documents/research/drone_sim_lab/assets/animals/crab/crab_direct.pth"


class PretrainedRlAntController(RlAntController):
    def __init__(self, parent_env: IsaacEnv, env_cfg=CrabEnvCfg()):
        super().__init__(parent_env, env_cfg)
        self.env_cfg = env_cfg
        from animals.ant.agent import Agent

        agent_cfg = load_cfg_from_registry(TASK_NAME, "rl_games_cfg_entry_point")

        self.agent = Agent(agent_cfg, self.observation_space, self.action_space, RESUME_PATH)

        self.current_step = 0
        self.last_obs = None

    def pre_decimation(self):
        action = self.agent.get_action(self.last_obs)
        self.actions = action.clone()

    def pre_step(self):
        if self.current_step % self.env_cfg.decimation == 0:
            self.pre_decimation()
        self.apply_action()

    def post_step(self):
        super().post_step()

        self.current_step += 1
        if self.current_step % self.env_cfg.decimation == 0:
            self.post_decimation()

    def post_decimation(self):
        # -- update env counters (used for curriculum generation)
        self.episode_length += 1  # step in current episode

        self.reset_terminated, self.reset_time_outs = self.get_dones()
        self.reset_buf = self.reset_terminated or self.reset_time_outs

        # -- reset env if terminated/timed-out and log the episode information
        if self.reset_buf: self.reset_idx()

        self.update_obs()

    def reset(self):
        # reset state of scene
        self.reset_idx()

        # return observations
        self.update_obs()
        self.agent.init(self.last_obs)
