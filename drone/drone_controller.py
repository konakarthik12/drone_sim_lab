import numpy as np
from pegasus.simulator.logic.backends import Backend
from scipy.spatial.transform import Rotation
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic import PegasusInterface
from drone.manipulators import Manipulators

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


class DroneController:
    def __init__(self, parent_env):
        log.info("Initializing Multirotor and Objects")

        self.world = parent_env.world
        self.pg = PegasusInterface()
        self.pg._world = self.world

        self._config_multirotor = MultirotorConfig()

        config_multirotor = MultirotorConfig()

        # There is a really annoying issue with previous px4 connections not being killed sometimes
        # This is a workaround for that
        force_kill_px4()
        self.backend = self.get_backend()
        config_multirotor.backends = [self.backend] if self.backend else []
        self.stage_prefix = "/World/drone"
        self.drone = Multirotor(
            self.stage_prefix,
            "/home/kkona/Documents/research/drone_sim_lab/assets/drones/iris_with_arm.usd",
            0,
            [0.0, 0.0, 1.02],
            list(Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat()),
            config=config_multirotor,
        )

        log.info("Initialized the Drone")

        self.manipulators = Manipulators(self.world)

    def get_backend(self) -> Backend | None:
        return None

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

