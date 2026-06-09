from pathlib import Path
import argparse

from spark_detector.detector import YOLOSparkDetector
from spark_detector.video_source import VideoFileSource
from spark_detector.output_handler import VideoFileOutput
from spark_detector.pipeline import VideoPipeline
from spark_detector.preprocessing import MaskPreprocessor


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-path",
        type=str,
        required=True,
        help="Путь к входному видеофайлу",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="Путь для сохранения выходного видеофайла",
    )

    parser.add_argument(
        "--use-preprocessor",
        type=bool,
        default=False,
        help="Использовать препроцессинг",
    )

    parser.add_argument(
        "--use-detector", type=bool, default=True, help="Использовать детекцию"
    )

    parser.add_argument(
        "-m",
        "--model-path",
        type=str,
        help="Путь к весам модели YOLO",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="'0' для GPU, 'cpu' для CPU (по умолчанию: '0')",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    input_path = Path(args.input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Входной файл не найден: '{input_path}'")

    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preprocessor = None
    if args.use_preprocessor:
        preprocessor = MaskPreprocessor()

    detector = None
    if args.use_detector:
        model_path = Path(args.model_path)

        if not model_path.is_file():
            raise FileNotFoundError(f"Модель не найдена: '{model_path}'")

        detector = YOLOSparkDetector(model_path=args.model_path)

    pipeline = VideoPipeline(detector=detector, preprocessor=preprocessor)

    source = VideoFileSource(video_path=input_path)
    output_handler = VideoFileOutput(
        output_path=output_path,
        fps=source.get_fps(),
        frame_size=source.get_frame_size(),
    )

    pipeline.process(source, output_handler)


if __name__ == "__main__":
    main()
