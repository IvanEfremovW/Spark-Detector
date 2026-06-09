from abc import ABC, abstractmethod
from numpy.typing import NDArray
import cv2


class BaseOutputHandler(ABC):
    @abstractmethod
    def write(self, frame: NDArray):
        pass

    @abstractmethod
    def close(self):
        pass


class VideoFileOutput(BaseOutputHandler):
    def __init__(self, output_path: str, fps: float, frame_size: tuple):
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ty: ignore[unresolved-attribute]
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    def write(self, frame: NDArray):
        self.writer.write(frame)

    def close(self):
        self.writer.release()
