import time

import numpy as np
from scipy.spatial.transform import Rotation

from utils import log

"""
Given a parent environment, this class creates a drone controller object that can be used to control the drone.
"""


class DroneControllerQGroundControl:
    def __init__(self, parent_env, init_pos=np.array([0, 0, 2.5])):
        from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
        from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
        log.info("Initializing Multirotor and Objects")

        self.world = parent_env.world
        self.pg = parent_env.pg
        self.init_pos = init_pos

        self._config_multirotor = MultirotorConfig()

        config_multirotor = MultirotorConfig()
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": True,
            "px4_dir": self.pg.px4_path,
            "px4_vehicle_model": self.pg.px4_default_airframe
        })
        config_multirotor.backends = [PX4MavlinkBackend(mavlink_config)]
        self.drone = Multirotor(
            "/World/quadrotor",
            "/home/kkona/Documents/research/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/assets/Robots/Iris/iris_copy.usd",
            0,
            [0.0, 0.0, 1.02],
            list(Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat()),
            config=config_multirotor,
        )

        log.info("Initialized the Drone")
        from drone.manipulators import Manipulators

        self.manipulators = Manipulators(self.world, self.drone)

    def reset(self, reset_pos=None):
        time.sleep(0.5)

    # action space (3,): [joint1_pos, joint2_pos, gripper_close]
    def step(self, action):
        self.manipulators.step(action)
