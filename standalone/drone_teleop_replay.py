from sim.app import init_app

init_app()

from pandas import read_pickle
from drone.drone_env import DroneEnv

env = DroneEnv(layout_type="grid")

commands = read_pickle("drone_commands.pkl")
print(len(commands))
commands = read_pickle("episode_actions.pkl")
env.reset()
for action in commands:
    env.step(action)
