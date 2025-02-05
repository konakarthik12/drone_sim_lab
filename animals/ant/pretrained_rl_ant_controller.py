from __future__ import annotations

from typing import TYPE_CHECKING

from omni.isaac.lab_tasks.utils import load_cfg_from_registry

from animals.ant.rl_ant_controller import RlAntController
from rl_games_helper.ant_env_cfg import AntEnvCfg

if TYPE_CHECKING:
    from rl_games_helper.locomotion_env import LocomotionEnv

TASK_NAME = "Isaac-Ant-Direct-v0"
RESUME_PATH = "/home/kkona/Documents/research/drone_sim_lab/assets/animals/ant_direct_policy.pth"


class PretrainedRlAntController(RlAntController):
    def __init__(self, parent_env: LocomotionEnv, env_cfg: AntEnvCfg):
        super().__init__(parent_env, env_cfg)
        self.env_cfg = env_cfg
        from animals.ant.rl.agent import Agent

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
        self.robot.update(self.env_cfg.sim.dt)

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
