from abc import ABC, abstractmethod
from typing import Iterator
from numpy.typing import NDArray


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
