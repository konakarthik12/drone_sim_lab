from omni.isaac.lab.app import AppLauncher

app = None


def init_app(headless=False):
    global app
    # TODO: Fix headless mode
    assert not headless, "Headless mode is not working for some reason"
    app_launcher = AppLauncher(headless=headless)
    app = app_launcher.app


def get_app():
    assert app is not None, "App is not initialized, called init_app() first"
    return app
