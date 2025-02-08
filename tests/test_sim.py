from sim.isaac_env import IsaacEnv

env = IsaacEnv()
app = env.app

while app.is_running():
    app.update()
