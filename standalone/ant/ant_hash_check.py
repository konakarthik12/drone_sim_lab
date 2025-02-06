from sim.app import init_app, get_app

init_app(headless=False)
app = get_app()

from torch import Tensor
from animals.ant.ant_env_cfg import AntEnvCfg
from animals.ant.ant_env import AntEnv

env_cfg = AntEnvCfg()
env = AntEnv(env_cfg)


def sha1_array(arr: Tensor):
    import hashlib

    from numpy import ndarray
    tensor_np: ndarray = arr.detach().cpu().numpy()
    tensor_bytes = tensor_np.tobytes()
    sha1_hash = hashlib.sha1(tensor_bytes).hexdigest()

    return sha1_hash


env.reset()

for _ in range(3):
    # decimation = 4
    for _ in range(4):
        env.step(None)
assert sha1_array(env.ant_controller.last_obs) == "3cbeb8f5a1e73b228b90ffdbca2a073b0557bedd"

env.close()
app.close()
