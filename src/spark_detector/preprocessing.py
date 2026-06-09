from abc import ABC, abstractmethod
from numpy.typing import NDArray
import numpy as np
import cv2


class BasePreprocessor(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def process(self, frame: NDArray) -> NDArray:
        pass


class MaskPreprocessor(BasePreprocessor):
    def __init__(
        self,
        brightness_threshold: int = 230,
        history: int = 500,
        var_threshold: int = 16,
    ):
        self.brightness_threshold = brightness_threshold
        self.history = history
        self.var_threshold = var_threshold
        self.bg_subtractor = None
        self.kernel = np.ones((3, 3), np.uint8)

    def reset(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history, varThreshold=self.var_threshold, detectShadows=False
        )

    def process(self, frame: NDArray) -> NDArray:
        if self.bg_subtractor is None:
            self.reset()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        _, bright_mask = cv2.threshold(
            gray, self.brightness_threshold, 255, cv2.THRESH_BINARY
        )
        motion_mask = self.bg_subtractor.apply(gray)
        combined_mask = cv2.bitwise_and(bright_mask, motion_mask)

        clean_mask = cv2.dilate(combined_mask, self.kernel, iterations=1)

        return cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
