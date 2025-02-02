import cv2
from omni.isaac.sensor import Camera


def convert_frames_to_video(images, output_path, fps=25):
    # images = images[40:]
    print(images[0].shape)
    height, width, _ = images[0].shape  # Get dimensions from the first image

    # Define the video writer
    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format

    # skip_seconds = 22
    # skip_frames = skip_seconds * fps
    # images = images[skip_frames:]
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write images to the video
    for img in images:
        video_writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # Convert RGB to BGR (OpenCV uses BGR format)

    # Release the video writer
    video_writer.release()


class VideoRecorder:
    def __init__(self, camera_path, fps):
        self.camera = Camera(camera_path, resolution=(1280, 720),
                             )
        self.frames = []
        self.fps = fps

    def capture(self):
        self.frames.append(self.camera.get_rgb())

    def save(self, output_path):
        convert_frames_to_video(self.frames, output_path, self.fps)
