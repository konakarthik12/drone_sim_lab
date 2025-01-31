from sim.app import init_app, get_app

init_app(headless=False)
app = get_app()
from ant_env_cfg import AntEnvCfg
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import load_cfg_from_registry
from rl_games_raw import RlGamesVecEnvWrapper

from locomotion_env import LocomotionEnv
task_name = "Isaac-Ant-Direct-v0"
resume_path = "/home/kkona/Documents/research/drone_sim_lab/assets/animals/ant_direct_policy.pth"

import omni.isaac.lab_tasks  # noqa: F401

"""Play with RL-Games agent."""

env_cfg = AntEnvCfg()
env_cfg.sim.device = "cpu"
env_cfg.scene.num_envs = 1
env_cfg.sim.use_fabric = False

# create isaac environment
isaac_env = LocomotionEnv(env_cfg)
from drone.drone_controller_qgroundcontrol import DroneControllerQGroundControl

# create drone controller
# drone_controller = DroneControllerQGroundControl(isaac_env)
# wrap around environment for rl-games
env = RlGamesVecEnvWrapper(isaac_env)


from agent import Agent
agent_cfg= load_cfg_from_registry(task_name, "rl_games_cfg_entry_point")
agent_cfg["params"]["config"]["device"] = "cpu"

agent = Agent(agent_cfg, resume_path)

# reset environment
obs = env.reset()
agent.init(obs)

# simulate environment
# note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
#   attempt to have complete control over environment stepping. However, this removes other
#   operations such as masking that is used for multi-agent learning by RL-Games.
while app.is_running():

    actions = agent.get_action(obs)
    # env stepping
    actions = env.pre_step(actions)
    # perform environment step
    obs_dict, _, _, _, _ = isaac_env.step(actions)
    obs = env.post_step(obs_dict)

# close the simulator
env.close()

# close sim app
app.close()
