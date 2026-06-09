from abc import ABC, abstractmethod
from typing import Iterator
from numpy.typing import NDArray
import cv2


class BaseVideoSource(ABC):
    @abstractmethod
    def frames(self) -> Iterator[NDArray]:
        pass

    @abstractmethod
    def get_fps(self) -> float:
        pass

    @abstractmethod
    def get_frame_size(self) -> tuple[int, int]:
        pass


class VideoFileSource(BaseVideoSource):
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frames(self) -> Iterator[NDArray]:
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

    def get_fps(self) -> float:
        return self.fps

    def get_frame_size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def __del__(self):
        if hasattr(self, "cap"):
            self.cap.release()
