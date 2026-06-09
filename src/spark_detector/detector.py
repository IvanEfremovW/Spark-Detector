from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from ultralytics import YOLO


class BaseSparkDetector(ABC):
    @abstractmethod
    def detect(self, frame: NDArray) -> NDArray:
        pass


class YOLOSparkDetector(BaseSparkDetector):
    def __init__(self, model_path: Path, device: int | str = "cuda"):
        self.model = YOLO(model_path)
        self.device = device

        # Warm-up
        dummy_frame = np.zeros((736, 736, 3), dtype=np.uint8)
        self.model.predict(dummy_frame, verbose=False, save=False)

    def detect(self, frame: NDArray) -> NDArray:
        results = self.model.predict(
            source=frame,
            verbose=False,
        )

        annotated_frame = results[0].plot()

        return annotated_frame
