## IsaacLab 

### Using the following training technique:
* Refer to this for RL training [link](https://isaac-sim.github.io/IsaacLab/v1.4.0/source/overview/reinforcement-learning/rl_existing_scripts.html#rl-games)
* **Note :** using Isaac Lab `v1.4.0`.
* Training on this IsaacLab env : `Isaac-Ant-Direct-v0`
```
# install python module (for rl-games)
./isaaclab.sh -i rl_games
# run script for training
./isaaclab.sh -p source/standalone/workflows/rl_games/train.py --task Isaac-Ant-Direct-v0 --headless
# run script for playing with 32 environments
./isaaclab.sh -p source/standalone/workflows/rl_games/play.py --task Isaac-Ant-Direct-v0 --num_envs 32 --checkpoint /PATH/TO/model.pth
# run script for recording video of a trained agent (requires installing `ffmpeg`)
./isaaclab.sh -p source/standalone/workflows/rl_games/play.py --task Isaac-Ant-Direct-v0 --headless --video --video_length 200
```

### Setting Up New Environment:
* Since we are using the Animal Agent Environment - We replicated Ant Env for Our Crab Env.
* Steps:
1. Copied `Ant` folder from this [directory](https://github.com/konakarthik12/IsaacLab/tree/crab-rl/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct) and save it with new name (`<name>`) in this directory. In our case we named it as `crab`.
2. Now rename and edit the `<name>_env.py` file inside the newly formed folder `<name>`. Like `robot prim path` etc, and import the `CFG` from the file created below in `lab_assets`.
3. In the `__init__.py` file register the env in gymasium. Example for `crab env`:
```
import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Crab-Direct-v0",
    entry_point=f"{__name__}.crab_env:CrabEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.crab_env:CrabEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CrabPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
```
4. Similarly create copy of the `ant.py` in this [directory](https://github.com/konakarthik12/IsaacLab/tree/crab-rl/source/extensions/omni.isaac.lab_assets/omni/isaac/lab_assets) and rename (with your defined name) and save it in the same directory.
5. Edit this copied `<name>.py` file, `CFG` details, like `usd path`, `init_state`  etc.
6. This `CFG` will be imported in the `env` created above.
7. Now this registered env (in `step 3`) can be used similarly for training as mentioned [above](#using-the-following-training-technique).

