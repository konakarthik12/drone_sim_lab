from sim.app import init_app
init_app()

from pandas import read_pickle
import numpy as np
from drone.drone_env import DroneEnv

env = DroneEnv(layout_type="grid")

commands = read_pickle("drone_commands.pkl")
print(len(commands))
env.reset()
for action in commands:
    action = np.append(action, [0.0] * 3)
    env.step(action)
