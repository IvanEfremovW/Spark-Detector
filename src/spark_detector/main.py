import os
import sys
import argparse

from spark_detector.detector import YOLOSparkDetector
from spark_detector.video_source import VideoFileSource
from spark_detector.output_handler import VideoFileOutput
from spark_detector.pipeline import VideoPipeline


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-path",
        type=str,
        required=True,
        help="Путь к входному видеофайлу.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=str,
        required=True,
        help="Путь для сохранения выходного видеофайла.",
    )

    parser.add_argument(
        "-m",
        "--model-path",
        type=str,
        required=True,
        default="models/yolov26s.pt",
        help="Путь к весам модели YOLO (по умолчанию: 'models/yolov26s.pt').",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="'0' для GPU, 'cpu' для CPU (по умолчанию: '0').",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    if not os.path.exists(args.input_path):
        print(f"Входной файл не найден: '{args.input_path}'")
        sys.exit(1)

    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    detector = YOLOSparkDetector(model_path=args.model_path)

    pipeline = VideoPipeline(detector)

    source = VideoFileSource(video_path=args.input_path)
    output_handler = VideoFileOutput(
        output_path=args.output_path,
        fps=source.get_fps(),
        frame_size=source.get_frame_size(),
    )

    pipeline.process(source, output_handler)


if __name__ == "__main__":
    main()
