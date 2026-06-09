from abc import ABC, abstractmethod

from .video_source import BaseVideoSource
from .output_handler import BaseOutputHandler
from .detector import BaseSparkDetector


class BasePipeline(ABC):
    @abstractmethod
    def process(self, source: BaseVideoSource, output_handler: BaseOutputHandler):
        pass


class VideoPipeline:
    def __init__(
        self,
        detector: BaseSparkDetector,
    ):
        self.detector = detector

    def process(
        self,
        source: BaseVideoSource,
        output_handler: BaseOutputHandler,
    ):
        try:
            for frame in source.frames():
                annotated_frame = self.detector.detect(frame)

                output_handler.write(annotated_frame)
        finally:
            output_handler.close()
