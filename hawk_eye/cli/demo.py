"""Run the DL video demo."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--court_model_path',
                        type=Path,
                        required=True)
    parser.add_argument('--ball_model_path',
                        type=Path,
                        required=True)
    parser.add_argument('--bounce_model_path',
                        type=Path,
                        required=True)
    parser.add_argument('--video_path',
                        type=Path,
                        required=True)

    args = parser.parse_args()

    from hawk_eye.pipelines.video_demo import video_demo
    video_demo(args)


if __name__ == '__main__':
    main()
