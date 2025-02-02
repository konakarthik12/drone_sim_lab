from pegasus.simulator.logic.backends import PX4MavlinkBackendConfig, PX4MavlinkBackend

from utils import save_pickle


class PX4MavlinkRecorderBackendConfig(PX4MavlinkBackendConfig):
    def __init__(self, config=None):
        super().__init__(config or {})

        self.record_path = self.config.get("commands_path")


class PX4MavlinkRecorderBackend(PX4MavlinkBackend):
    config: PX4MavlinkRecorderBackendConfig

    def __init__(self, config: PX4MavlinkRecorderBackendConfig = None):
        super().__init__(config)

        self.commands = []

    def update(self, dt):
        super().update(dt)
        self.commands.append(self.input_reference().copy())

    def stop(self):
        super().stop()
        save_pickle(self.config.record_path, self.commands)
        print("Commands saved to", self.config.record_path)
