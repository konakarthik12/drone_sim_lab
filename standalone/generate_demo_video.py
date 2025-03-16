import pickle
import cv2
import numpy as np


def load_pickle_images(pickle_file):
    """Loads images from a pickle file, removes the first frame, and returns images, height, and width."""
    with open(pickle_file, "rb") as f:
        images = pickle.load(f)

    images = images[1:]  # Remove the first frame
    height, width, _ = images[0].shape  # Extract dimensions

    return images, height, width


# Load images from both pickle files
images_1, height_1, width_1 = load_pickle_images("episode_frames.pkl")
images_2, height_2, width_2 = load_pickle_images("episode_frames_ui.pkl")

# Ensure both lists have the same number of frames and resolution
assert len(images_1) == len(images_2), "Frame lists must have the same length"
assert (height_1, width_1) == (height_2, width_2), "Frame dimensions must match"

# Define video writer (double the width for side-by-side effect)
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
fps = 30
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width_1 * 2, height_1))

# Process and write frames
for img1, img2 in zip(images_1, images_2):
    img1_bgr = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    img2_bgr = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR

    # Concatenate images horizontally
    combined_frame = np.hstack((img1_bgr, img2_bgr))

    # Write to video
    video_writer.write(combined_frame)

# Release video writer
video_writer.release()
print(f"Side-by-side video saved as {output_path}")
