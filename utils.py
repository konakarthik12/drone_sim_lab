import logging

from carb.input import GamepadEvent


def enable_gpu_dynamics():
    from omni.physx import acquire_physx_interface
    physx_interface = acquire_physx_interface()
    physx_interface.overwrite_gpu_setting(1)


def add_gamepad_callback(callback, gamepad_id=0):
    import carb.settings
    import carb.input
    import omni.appwindow

    carb_settings_iface = carb.settings.get_settings()
    carb_settings_iface.set_bool("/persistent/app/omniverse/gamepadCameraControl", False)
    carb_input = carb.input.acquire_input_interface()
    app_window = omni.appwindow.get_default_app_window()

    gamepad = app_window.get_gamepad(gamepad_id)

    gamepad_sub = carb_input.subscribe_to_gamepad_events(
        gamepad,
        callback
    )
    return gamepad_sub


class GamepadButtonPressWatcher:
    def __init__(self, button):
        self.pressed = False
        self.button = button

    def gamepad_callback(self, event: GamepadEvent):
        if event.input == self.button and event.value > 0.5:
            self.pressed = True
            return False
        return True


def set_active_camera(camera_path):
    from omni.kit.viewport.utility import get_active_viewport
    viewport = get_active_viewport()
    viewport.set_active_camera(camera_path)


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
        msg = self.parse_message(*args)
        self.log.info(msg)

    def debug(self, *args):
        msg = self.parse_message(*args)
        self.log.debug(msg)


log = Logger("drone_sim")
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def relative_file(path):
    return os.path.join(ROOT_DIR, path)


def open_pickle(path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path, data):
    import pickle
    with open(path, "wb") as f:
        # noinspection PyTypeChecker
        pickle.dump(data, f)
