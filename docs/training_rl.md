## IsaacLab: Reinforcement Learning Training Guide

This guide explains how to train reinforcement learning (RL) agents in IsaacLab, and how to set up a new custom environment (e.g., for a new robot).

---

### 1. RL Training with IsaacLab

- **Reference:** See the [IsaacLab RL training documentation](https://isaac-sim.github.io/IsaacLab/v1.4.0/source/overview/reinforcement-learning/rl_existing_scripts.html#rl-games) for more details.
- **Version:** This guide uses IsaacLab `v1.4.0`.
- **Default Environment:** Example shown for `Isaac-Ant-Direct-v0`.

**Common Training Commands:**
```bash
# Install the rl-games Python module
./isaaclab.sh -i rl_games

# Train an agent in headless mode
./isaaclab.sh -p source/standalone/workflows/rl_games/train.py --task Isaac-Ant-Direct-v0 --headless

# Play with a trained agent using 32 parallel environments
./isaaclab.sh -p source/standalone/workflows/rl_games/play.py --task Isaac-Ant-Direct-v0 --num_envs 32 --checkpoint /PATH/TO/model.pth

# Record a video of a trained agent (requires ffmpeg)
./isaaclab.sh -p source/standalone/workflows/rl_games/play.py --task Isaac-Ant-Direct-v0 --headless --video --video_length 200
```

---

### 2. Creating a New Custom Environment

Suppose you want to create a new environment (e.g., for a "Crab" robot) by replicating and modifying the existing Ant environment.

#### **Step-by-Step Instructions:**

1. **Copy the Ant Environment Folder:**
   - Go to the [Ant environment directory](https://github.com/konakarthik12/IsaacLab/tree/crab-rl/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct).
   - Copy the `Ant` folder and rename it to your new environment name (e.g., `crab`).

2. **Modify the Environment Python File:**
   - Inside your new folder, rename and edit `<name>_env.py` (e.g., `crab_env.py`).
   - Update details such as the robot's prim path and import the correct configuration (`CFG`) from your asset file.

3. **Register the Environment in Gymnasium:**
   - Edit the `__init__.py` in your new environment folder.
   - Register your environment with Gymnasium. Example for the Crab environment:
     ```python
     import gymnasium as gym
     from . import agents

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

4. **Create and Configure the Asset File:**
   - Copy `ant.py` from the [lab_assets directory](https://github.com/konakarthik12/IsaacLab/tree/crab-rl/source/extensions/omni.isaac.lab_assets/omni/isaac/lab_assets).
   - Rename it (e.g., `crab.py`) and update the configuration (`CFG`) for your robot (e.g., USD path, initial state, etc.).

5. **Connect the Asset Configuration:**
   - Ensure your environment imports and uses the correct `CFG` from your new asset file.

6. **Train with Your New Environment:**
   - You can now use the same training commands as above, replacing the task name with your new environment (e.g., `Isaac-Crab-Direct-v0`).

---

**Summary:**  
By following these steps, you can quickly duplicate and adapt an existing IsaacLab RL environment for a new robot or agent, and train it using the provided RL