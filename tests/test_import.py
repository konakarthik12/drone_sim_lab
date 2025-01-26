from sim.isaac_env import IsaacEnv

env = IsaacEnv(headless=False)

import omni.isaac.kit
import carb
import omni.isaac.lab.sim
import pegasus.simulator.logic
import omni.isaac.version

modules = [
    omni.isaac.kit,
    carb,
    omni.isaac.lab.sim,
    pegasus.simulator.logic,
    omni.isaac.dynamic_control,
    omni.isaac.version

]
for module in modules:
    print(module.__name__, module.__file__)
