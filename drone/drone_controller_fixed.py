import numpy as np
from omni.isaac.lab.sim import SimulationContext
from scipy.spatial.transform import Rotation

from utils import log
from sim.dc_interface import dc


class DroneControllerFixed:
    def __init__(self, parent_env):
        from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
        from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
        log.info("Initializing Multirotor and Objects")

        self.world: SimulationContext = parent_env.world
        from pegasus.simulator.logic import PegasusInterface
        self.pg = PegasusInterface()
        self.pg._world = self.world
        self.init_pos = [0.0, 0.0, 0.4]
        self.init_rot = list(Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat())
        self._config_multirotor = MultirotorConfig()

        config_multirotor = MultirotorConfig()


        config_multirotor.backends = []
        self.stage_prefix = "/World/drone"
        self.drone = Multirotor(
            self.stage_prefix,
            "/home/kkona/Documents/research/drone_sim_lab/assets/drones/iris_with_arm.usd",
            0,
            self.init_pos,
            self.init_rot,
            config=config_multirotor,
        )

        from omni.physx.scripts import utils
        import omni.isaac.core.utils.prims as prims_utils
        from_prim = prims_utils.get_prim_at_path("/World")
        to_prim = prims_utils.get_prim_at_path("/World/drone/body")

        utils.createJoint(self.world.stage, "Fixed", from_prim, to_prim)

        log.info("Initialized the Drone")
        from drone.manipulators import Manipulators


        self.manipulators = Manipulators(self.world)

    def reset(self, reset_pos=None):
        pass

    # action space (3,): [joint1_pos, joint2_pos, gripper_close]
    def step(self, action):
        self.manipulators.step(action)
        # dc.set_rigid_body_pose(self.drone_body, self.init_transform)
        return
    def post_init(self):
        self.drone_articulation = dc.get_articulation(self.stage_prefix)
        self.drone_body = dc.get_rigid_body("/World/drone/body")

        self.manipulators.post_init(self.drone_articulation)
        return