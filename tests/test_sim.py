from simulator.isaac_env import IsaacEnv

env = IsaacEnv(headless=False, layout_type="air")
app = env.app

while app.is_running():
    app.update()
