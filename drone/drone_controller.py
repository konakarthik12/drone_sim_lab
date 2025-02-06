import numpy as np
from pegasus.simulator.logic import PegasusInterface
from pegasus.simulator.logic.backends import BackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from scipy.spatial.transform import Rotation

from drone.drone_utils.pegasus_backend import PegasusBackend
from drone.manipulators import Manipulators
from sim.controller import Controller


class DirectBackend(PegasusBackend):
    def __init__(self, config=BackendConfig()):
        super().__init__(config)
        self.last_action = np.zeros(4)

    def input_reference(self):
        return self.last_action


"""
Given a parent environment, this class creates a drone controller object that can be used to control the drone.
"""


def force_kill_px4():
    import psutil
    for proc in psutil.process_iter():
        # check whether the process name matches
        if proc.name() == "px4":
            proc.kill()


class DroneController(Controller):
    backend: DirectBackend

    def __init__(self, parent_env):
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
        init_pos = np.array([0.0, 0.0, 1.02])
        init_rot = np.array([0.0, 0.0, 0])
        # noinspection PyTypeChecker
        self.drone = Multirotor(
            self.stage_prefix,
            "/home/kkona/Documents/research/drone_sim_lab/assets/drones/iris_with_arm.usd",
            0,
            init_pos,
            Rotation.from_euler("XYZ", init_rot, degrees=True).as_quat(),
            config=config_multirotor,
        )

        self.manipulators = Manipulators(self.world, self.stage_prefix)

    def get_backend(self):
        return DirectBackend()

    def reset(self, reset_pos=None):
        pass

    # action space (7,): [motor1, ..., motor4, joint1_pos, joint2_pos, gripper_close]
    def step(self, action):
        self.backend.last_action = action[:4]
        self.manipulators.step(action[4:])
        return

    def post_init(self):
        self.manipulators.post_init()

    def close(self):
        self.backend.stop()
