import cv2
from omni.isaac.sensor import Camera

from utils import save_pickle


class VideoRecorder:
    def __init__(self, camera_path, fps):
        self.camera = Camera(camera_path, resolution=(1280, 720), )
        self.frames = []
        self.fps = fps

    def init(self):
        self.camera.initialize()

    def step(self):
        self.frames.append(self.camera.get_rgb())

    def save(self, output_path):
        save_pickle(output_path, self.frames)
