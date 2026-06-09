from abc import ABC, abstractmethod
from numpy.typing import NDArray


class BaseSparkDetector(ABC):
    @abstractmethod
    def detect(self, frame: NDArray) -> NDArray:
        pass
