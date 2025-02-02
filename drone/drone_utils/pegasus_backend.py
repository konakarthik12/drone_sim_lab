from pegasus.simulator.logic.backends import Backend, BackendConfig


# By default Pegasus forces the implementation of all of these abstract methods
# so this helper just fill them in with empty bodies
class PegasusBackend(Backend):
    def __init__(self, config=BackendConfig()):
        super().__init__(config)

    def update_sensor(self, sensor_type: str, data):
        pass

    def update_graphical_sensor(self, sensor_type: str, data):
        pass

    def update_state(self, state):
        pass

    def input_reference(self):
        return [0.0 for _ in range(4)]

    def update(self, dt: float):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def reset(self):
        pass
