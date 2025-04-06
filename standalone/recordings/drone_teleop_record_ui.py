from sim.app import init_app
from utils import save_pickle, get_active_camera, enable_gamepad_control

init_app()

from pandas import read_pickle
from drone.drone_env import DroneEnv
from omni.isaac.sensor import Camera

env = DroneEnv(layout="grid")

commands = read_pickle("output/episode_actions.pkl")

camera_path = get_active_camera()
camera = Camera(camera_path, resolution=(1280, 720))
enable_gamepad_control()

env.reset()
camera.initialize()
frames = []
count = 0
for action in commands:
    env.step(action)
    if count % 4 == 0:
        env.sim.render()
        frames.append(camera.get_rgb())
    count += 1

frames_path = "output/episode_frames_ui.pkl"
save_pickle(frames_path, frames)
print(f"Frames saved to {frames_path}")
