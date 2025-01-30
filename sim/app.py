from omni.isaac.lab.app import AppLauncher

_app = None


def init_app(headless=False):
    global _app

    # TODO: Fix headless mode
    assert not headless, "Headless mode is not working for some reason"
    app_launcher = AppLauncher(headless=headless)
    _app = app_launcher.app
    return _app


def get_app():
    global _app
    assert _app is not None, "App is not initialized, called init_app() first"
    return _app
