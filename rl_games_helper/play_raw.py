from omni.isaac.lab.app import AppLauncher

app_launcher = AppLauncher()
simulation_app = app_launcher.app
"""Rest everything follows."""

from drone.drone_controller_qgroundcontrol import DroneControllerQGroundControl

from omni.isaac.lab_tasks.direct.ant.ant_env import AntEnv, AntEnvCfg
from omni.isaac.lab_tasks.direct.locomotion.locomotion_env import LocomotionEnv

import torch

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from rl_games_raw import RlGamesVecEnvWrapper

from rl_games_helper import Agent
from typing import cast

task_name = "Isaac-Ant-Direct-v0"

"""Play with RL-Games agent."""
# parse env configuration
env_cfg = cast(AntEnvCfg, parse_env_cfg(task_name, device="cpu", num_envs=1, use_fabric=False))
agent_cfg = load_cfg_from_registry(task_name, "rl_games_cfg_entry_point")

resume_path = "/home/kkona/Documents/research/drone_sim_lab/rl_games_helper/ant_direct_policy.pth"

env_cfg.sim.device = "cpu"


# create isaac environment
isaac_env = LocomotionEnv(env_cfg)
print(isaac_env.__class__.__name__)

# wrap around environment for rl-games
env = RlGamesVecEnvWrapper(isaac_env)


# agent = get_rl_games_agent(agent_cfg, resume_path)

agent = Agent(agent_cfg, resume_path)

# reset environment
obs = env.reset()
agent.init(obs)

# simulate environment
# note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
#   attempt to have complete control over environment stepping. However, this removes other
#   operations such as masking that is used for multi-agent learning by RL-Games.
while simulation_app.is_running():

    actions = agent.get_action(obs)
    # env stepping
    actions = env.pre_step(actions)
    # perform environment step
    obs_dict, _, _, _, _ = isaac_env.step(actions)
    obs = env.post_step(obs_dict)

# close the simulator
env.close()

# close sim app
simulation_app.close()
