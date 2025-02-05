
from sim.app import init_app, get_app

init_app(headless=False)
app = get_app()

from utils import add_gamepad_callback, set_active_camera
from torch import Tensor
from ant_env_cfg import AntEnvCfg
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import load_cfg_from_registry
from drone.manipulators import ManipulatorState


task_name = "Isaac-Ant-Direct-v0"
resume_path = "/home/kkona/Documents/research/drone_sim_lab/assets/animals/ant_direct_policy.pth"

import omni.isaac.lab_tasks  # noqa: F401

"""Play with RL-Games agent."""

env_cfg = AntEnvCfg()
from rl_games_helper.locomotion_env import LocomotionEnv

# create isaac environment
env = LocomotionEnv(env_cfg)


# create drone controller
# drone_controller = DroneControllerQGroundControl(env)


def sha1_array(arr: Tensor):
    import hashlib

    # Ensure tensor is in CPU memory and convert to NumPy
    from numpy import ndarray
    tensor_np: ndarray = arr.detach().cpu().numpy()

    # Convert tensor to bytes
    tensor_bytes = tensor_np.tobytes()

    # Compute SHA-1 hash
    sha1_hash = hashlib.sha1(tensor_bytes).hexdigest()

    return sha1_hash

from animals.ant.rl.agent import Agent
agent_cfg = load_cfg_from_registry(task_name, "rl_games_cfg_entry_point")

agent_cfg["params"]["config"]["device"] = "cpu"
agent_cfg["params"]["config"]["device_name"] = "cpu"
set_active_camera("/World/drone/arm/arm_base/Camera")
agent = Agent(env.ant_controller, agent_cfg, resume_path)
mani_state = ManipulatorState()
add_gamepad_callback(mani_state.gamepad_callback)
# reset environment
env.reset()
obs = env.last_obs
# drone_controller.post_init()
agent.init(obs)
# simulate environment
# note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
#   attempt to have complete control over environment stepping. However, this removes other
#   operations such as masking that is used for multi-agent learning by RL-Games.
count = 0

while app.is_running():

    actions = agent.get_action(obs)
    # drone_controller.step(mani_state.as_action())
    count += 1
    if count ==4:
        assert sha1_array(obs) == "3cbeb8f5a1e73b228b90ffdbca2a073b0557bedd"
    for _ in range(4):
        env.step(actions)
    obs = env.last_obs

last_sha = sha1_array(obs)
# close the simulator
env.close()

# print("Last sha", last_sha)
# assert last_sha == "3cbeb8f5a1e73b228b90ffdbca2a073b0557bedd"
# close sim app
app.close()
