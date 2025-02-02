from pegasus.simulator.logic.backends import PX4MavlinkBackendConfig, PX4MavlinkBackend

from drone.drone_controller import DroneController


class DroneControllerQGroundControl(DroneController):

    def get_backend(self):
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": True,  # Launch PX4 automatically
            "px4_dir": self.pg.px4_path,
            "px4_vehicle_model": self.pg.px4_default_airframe
        })
        return PX4MavlinkBackend(mavlink_config)
