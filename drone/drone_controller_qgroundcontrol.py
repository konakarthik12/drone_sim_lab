from pegasus.simulator.logic.backends import PX4MavlinkBackend, PX4MavlinkBackendConfig

from drone.drone_controller import DroneController


class DroneControllerQGroundControl(DroneController):

    def __init__(self, parent_env):
        super().__init__(parent_env)

    def get_backend(self):
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": True,  # Launch PX4 automatically
            "px4_dir": self.pg.px4_path,
            "px4_vehicle_model": self.pg.px4_default_airframe,
        })
        return PX4MavlinkBackend(mavlink_config)

    def step(self, action):
        self.manipulators.step(action)

    def close(self):
        self.backend.stop()
