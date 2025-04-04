import pickle

import cv2

# Load the pickle file
with open("episode_frames.pkl", "rb") as f:
    images = pickle.load(f)
# Remove the first element (it's a bad image)
images = images[1:]

height, width, _ = images[0].shape  # Get dimensions from an image

# Define the video writer
output_path = "fpv_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
fps = 30
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Write images to the video
for img in images:
    video_writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR (OpenCV uses BGR format)

# Release the video writer
video_writer.release()
print(f"Video saved as {output_path}")
