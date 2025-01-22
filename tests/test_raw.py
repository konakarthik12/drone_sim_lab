from isaacsim import SimulationApp

app = SimulationApp({"headless": False})

while app.is_running():
    app.update()
