from sim.app import init_app

init_app()

from pandas import read_pickle
from drone.drone_env import DroneEnv
from omni.isaac.sensor import Camera
hdf5_path = "output/episode_frames.h5"

env = DroneEnv(layout="grid")

commands = read_pickle("output/episode_actions.pkl")
camera_path = "/World/drone/arm/arm_base/Camera"
camera = Camera(camera_path, resolution=(1920, 1080))
import h5py
from tqdm.auto import tqdm

env.reset()
camera.initialize()

env.step(commands[0])
env.sim.render()
frame = camera.get_rgb()
assert frame.shape == (0,)  # First frame is always bad
env.step(commands[1])
env.sim.render()
frame = camera.get_rgb()
assert frame.shape != (0,)
commands = commands[2:]  # First two actions already done
dataset_initialized = False

frame_shape = frame.shape
print(f"Frame shape: {frame_shape}")
with h5py.File(hdf5_path, "w") as f:
    # Create a resizable dataset with compression enabled
    dset = f.create_dataset(
        "frames",
        shape=(0, *frame_shape),
        maxshape=(None, *frame_shape),
        dtype=frame.dtype,
        chunks=(1, *frame_shape),  # or tune chunk size for your usage
        compression="gzip",
        compression_opts=4  # adjust compression level (1=fast, 9=small)
    )
    frame_idx = 0
    for i, action in enumerate(tqdm(commands)):
        env.step(action)
        if i % 2 == 0:
            env.sim.render()
            frame = camera.get_rgb()
            # Increase dataset size by one and add the new frame
            dset.resize(frame_idx + 1, axis=0)
            dset[frame_idx] = frame
            frame_idx += 1

print(f"Frames saved to {hdf5_path}")