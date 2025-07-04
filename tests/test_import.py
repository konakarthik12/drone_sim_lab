from sim.app import init_app, get_app
from sim.isaac_env import IsaacEnv
init_app()
env = IsaacEnv()
from sim.dc_interface import dc

import omni.isaac.kit
import carb.settings
import omni.isaac.lab.sim
import pegasus.simulator.logic
import omni.isaac.version
import omni.isaac.dynamic_control
import omni.appwindow
import omni.timeline
import omni.kit.viewport.utility
import omni.isaac.sensor
modules = [
    omni.isaac.kit,
    carb,
    carb.settings,
    omni.isaac.lab.sim,
    pegasus.simulator.logic,
    omni.isaac.dynamic_control,
    omni.isaac.version,
    omni.timeline,
    omni.kit.viewport.utility,
    omni.isaac.sensor
]
print(omni.appwindow.__file__)
print(dc.__file__)

for module in modules:
    print(module.__name__, module.__file__)

app = get_app()
app.close()