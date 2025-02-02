from sim.app import init_app

init_app()

from sim.video_recorder import VideoRecorder
from pandas import read_pickle
from drone.drone_env import DroneEnv

env = DroneEnv(layout_type="grid")

commands = read_pickle("episode_actions.pkl")
video_recorder = VideoRecorder("/World/drone/arm/arm_base/Camera", fps=30)
env.reset()
for action in commands:
    env.step(action)

video_path = "episode.mp4"
video_recorder.save(video_path)
print("Video saved to", video_path)
