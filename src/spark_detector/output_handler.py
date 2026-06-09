from abc import ABC, abstractmethod
from numpy.typing import NDArray


class BaseOutputHandler(ABC):
    @abstractmethod
    def write(self, frame: NDArray):
        pass

    @abstractmethod
    def close(self):
        pass
