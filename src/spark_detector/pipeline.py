from abc import ABC, abstractmethod

from .video_source import BaseVideoSource
from .output_handler import BaseOutputHandler


class BasePipeline(ABC):
    @abstractmethod
    def process(self, source: BaseVideoSource, output_handler: BaseOutputHandler):
        pass
