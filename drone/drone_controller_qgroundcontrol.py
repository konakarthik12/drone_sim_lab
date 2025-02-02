from drone.drone_controller import DroneController
from drone.mavlink_record import PX4MavlinkRecorderBackendConfig, PX4MavlinkRecorderBackend
from pegasus.simulator.logic.backends import PX4MavlinkBackend


class DroneControllerQGroundControl(DroneController):

    def __init__(self, parent_env, record_path=None):
        self.record_path = record_path
        super().__init__(parent_env)

    def get_backend(self):
        mavlink_config = PX4MavlinkRecorderBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": True,  # Launch PX4 automatically
            "px4_dir": self.pg.px4_path,
            "px4_vehicle_model": self.pg.px4_default_airframe,
            "commands_path": self.record_path
        })
        if self.record_path is None:
            return PX4MavlinkBackend(mavlink_config)
        else:
            return PX4MavlinkRecorderBackend(mavlink_config)

    def step(self, action):
        self.manipulators.step(action)

    def close(self):
        self.backend.stop()
