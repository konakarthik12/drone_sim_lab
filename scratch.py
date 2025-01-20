from sim.isaac_env import IsaacEnv

env = IsaacEnv(headless=False, layout_type="grid")
app = env.app

while app.is_running():
    app.update()
