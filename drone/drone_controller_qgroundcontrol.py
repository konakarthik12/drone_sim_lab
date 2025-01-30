import numpy as np
from scipy.spatial.transform import Rotation

from utils import log

"""
Given a parent environment, this class creates a drone controller object that can be used to control the drone.
"""

def force_kill_px4():
    import psutil
    for proc in psutil.process_iter():
        # check whether the process name matches
        if proc.name() == "px4":
            proc.kill()

class DroneControllerQGroundControl:
    def __init__(self, parent_env, init_pos=np.array([0, 0, 2.5])):
        from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
        from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
        log.info("Initializing Multirotor and Objects")

        self.world = parent_env.world
        from pegasus.simulator.logic import PegasusInterface
        self.pg = PegasusInterface()
        self.pg._world = self.world
        self.init_pos = init_pos

        self._config_multirotor = MultirotorConfig()

        config_multirotor = MultirotorConfig()

        # There is a really annoying issue with previous px4 connections not being killed sometimes
        # This is a workaround for that
        force_kill_px4()
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": True, # Launch PX4 automatically
            "px4_dir": self.pg.px4_path,
            "px4_vehicle_model": self.pg.px4_default_airframe
        })
        config_multirotor.backends = [PX4MavlinkBackend(mavlink_config)]
        self.stage_prefix = "/World/quadrotor"
        self.drone = Multirotor(
            self.stage_prefix,
            "/home/kkona/Documents/research/drone_sim_lab/assets/drones/iris_with_arm.usd",
            0,
            [0.0, 0.0, 1.02],
            list(Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat()),
            config=config_multirotor,
        )

        log.info("Initialized the Drone")
        from drone.manipulators import Manipulators


        self.manipulators = Manipulators(self.world)

    def reset(self, reset_pos=None):
        pass

    # action space (3,): [joint1_pos, joint2_pos, gripper_close]
    def step(self, action):
        self.manipulators.step(action)
        return
    def post_init(self):
        from sim.dc_interface import dc
        drone_articulation = dc.get_articulation(self.stage_prefix)

        self.manipulators.post_init(drone_articulation)
        return