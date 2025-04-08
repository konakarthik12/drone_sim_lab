from sim.app import init_app
init_app()
from sim.isaac_env import IsaacEnv

env = IsaacEnv()
app = env.app

while app.is_running():
    app.update()
