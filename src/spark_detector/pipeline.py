from abc import ABC, abstractmethod

from spark_detector.video_source import BaseVideoSource
from spark_detector.output_handler import BaseOutputHandler
from spark_detector.detector import BaseSparkDetector
from spark_detector.preprocessing import BasePreprocessor


class BasePipeline(ABC):
    @abstractmethod
    def process(self, source: BaseVideoSource, output_handler: BaseOutputHandler):
        pass


class VideoPipeline:
    def __init__(
        self, detector: BaseSparkDetector | None, preprocessor: BasePreprocessor | None
    ):
        self.detector = detector
        self.preprocessor = preprocessor

    def process(
        self,
        source: BaseVideoSource,
        output_handler: BaseOutputHandler,
    ):
        if self.preprocessor:
            self.preprocessor.reset()

        try:
            for frame in source.frames():
                processed_frame = frame

                if self.preprocessor:
                    processed_frame = self.preprocessor.process(processed_frame)

                if self.detector:
                    processed_frame = self.detector.detect(processed_frame)

                output_handler.write(processed_frame)
        finally:
            output_handler.close()
