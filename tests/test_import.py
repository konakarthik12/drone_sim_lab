from sim.app import init_app, get_app
from sim.isaac_env import IsaacEnv
init_app()
env = IsaacEnv()

import omni.isaac.kit
import carb.settings
import omni.isaac.lab.sim
import pegasus.simulator.logic
import omni.isaac.version
import omni.isaac.dynamic_control
import omni.appwindow
import omni.timeline
import omni.kit.viewport.utility
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


]
print(omni.appwindow.__file__)

for module in modules:
    print(module.__name__, module.__file__)

app = get_app()
app.close()