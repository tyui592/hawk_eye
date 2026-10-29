"""Create a reviewed court dataset copy."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('data/TennisCourtDetector'))
    parser.add_argument('--results', type=Path,
                        default=Path('data/outputs/court_detection/classical-v3/tenniscourt-val/results.json'))
    parser.add_argument('--destination', type=Path,
                        default=Path('data/TennisCourtDetector_v3_reviewed'))
    args = parser.parse_args()

    from hawk_eye.datasets.prepare_court import prepare_dataset
    try:
        prepare_dataset(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
