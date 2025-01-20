import time

import numpy as np
from scipy.spatial.transform import Rotation

from utils import log

"""
Given a parent environment, this class creates a drone controller object that can be used to control the drone.
"""


class DroneController:
    def __init__(self, parent_env, init_pos=np.array([0, 0, 2.5])):
        from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
        from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
        log.info("Initializing Multirotor and Objects")

        self.world = parent_env.world
        self.pg = parent_env.pg
        self.init_pos = init_pos

        self._config_multirotor = MultirotorConfig()

        # from pegasus.simulator.logic.thrusters import QuadraticThrustCurve
        # self._config_multirotor.thrust_curve = QuadraticThrustCurve(config={"rot_dir": [-1, 1, -1, 1]})

        # self._config_multirotor.backends = [
        #     fast_nonlinear_controller.FastNonLinearController(Kr=[2.0, 2.0, 2.0])
        # ]

        # drone_name = "cross_drone13_v10_mani_gripper"
        # drone_file = ROBOTS_ASSETS + f"/drone_models/{drone_name}.usd"
        # assert os.path.exists(drone_file), f"Drone file {drone_file} does not exist"
        config_multirotor = MultirotorConfig()
        # Create the multirotor configuration
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": True,
            "px4_dir": self.pg.px4_path,
            "px4_vehicle_model": self.pg.px4_default_airframe
            # CHANGE this line to 'iris' if using PX4 version bellow v1.14
        })
        config_multirotor.backends = [PX4MavlinkBackend(mavlink_config)]
        self.drone = Multirotor(
            "/World/quadrotor",
            "/home/kkona/Documents/research/PegasusSimulator/extensions/pegasus.simulator/pegasus/simulator/assets/Robots/Iris/iris_copy.usd",
            0,
            [0.0, 0.0, 1.02],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

        log.info("Initialized the Drone")
        # log.info("Drone Position: ", self.drone.get_body_pose())

    def reset(self, reset_pos=None):
        time.sleep(0.5)



    def step(self, action):
        pass
