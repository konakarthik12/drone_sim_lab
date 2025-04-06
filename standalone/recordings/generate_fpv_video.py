import cv2
import h5py
from tqdm.auto import tqdm
# Constants
H5_FILE = "output/episode_frames.h5"
FINAL_VIDEO = "output/fpv_video.mp4"
SIZE_LIMIT_MB = 10
FPS = 60

# Open the HDF5 file
with h5py.File(H5_FILE, "r") as f:
    frames = f["frames"]
    num_frames = frames.shape[0]
    height, width, _ = frames.shape[1:]

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(FINAL_VIDEO, fourcc, FPS, (width, height))

    # Write frames with progress bar
    for i in tqdm(range(num_frames), desc="Writing video"):
        frame = frames[i]  # stream one frame at a time
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)

    video_writer.release()

print(f"Video saved as {FINAL_VIDEO}")


# # Calculate target bitrate
# duration_sec = len(images) / FPS
# target_size_bits = SIZE_LIMIT_MB * 8 * 1024 * 1024 * 0.95
# video_bitrate = int(target_size_bits / duration_sec)
#
# # Clamp bitrate to a minimum to avoid ffmpeg issues
# video_bitrate = max(video_bitrate, 100_000)
#
# print(f"Target video bitrate: {video_bitrate} bps")
#
# # FFmpeg Two-Pass (slow preset, no audio)
# # First pass
# subprocess.run([
#     "ffmpeg", "-y", "-i", temp_video_path,
#     "-c:v", "libx264", "-preset", "slow", "-b:v", f"{video_bitrate}",
#     "-pass", "1", "-an", "-f", "mp4", os.devnull
# ])
#
# # Second pass
# subprocess.run([
#     "ffmpeg", "-y", "-i", temp_video_path,
#     "-c:v", "libx264", "-preset", "slow", "-b:v", f"{video_bitrate}",
#     "-pass", "2", "-an",
#     FINAL_VIDEO
# ])
#
# # Cleanup
# os.remove("ffmpeg2pass-0.log")
# os.remove(temp_video_path)
#
# print(f"Final video saved as {FINAL_VIDEO}")
