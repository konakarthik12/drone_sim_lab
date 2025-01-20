import logging


def enable_gpu_dynamics():
    from omni.physx import acquire_physx_interface
    physx_interface = acquire_physx_interface()
    physx_interface.overwrite_gpu_setting(1)


def add_gamepad_callback(callback, gamepad_id = 0):
    import carb
    import omni.kit
    carb_settings_iface = carb.settings.get_settings()
    carb_settings_iface.set_bool("/persistent/app/omniverse/gamepadCameraControl", False)
    carb_input = carb.input.acquire_input_interface()
    appwindow = omni.appwindow.get_default_app_window()

    gamepad = appwindow.get_gamepad(gamepad_id)

    gamepad_sub = carb_input.subscribe_to_gamepad_events(
        gamepad,
        # lambda event, *args,: self._on_gamepad_event(event, *args),
        callback
    )
    return gamepad_sub

class Logger:
    def __init__(self, name):
        self.log = logging.getLogger(name)
        self.log.setLevel(logging.WARN)

    def set_level(self, level):
        self.log.setLevel(level)

    def parse_message(self, *args):
        return " ".join(map(str, args))

    def error(self, *args):
        msg = self.parse_message(*args)
        self.log.error(msg)

    def info(self, *args):
        return
        msg = self.parse_message(*args)
        self.log.info(msg)

    def debug(self, *args):
        return
        msg = self.parse_message(*args)
        self.log.debug(msg)


log = Logger("drone_sim")
