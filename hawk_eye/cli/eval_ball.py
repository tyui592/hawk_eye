"""Run ball from the command line."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=Path, default=Path('./data/TrackNet/'))
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_dir', type=Path, default=None)
    parser.add_argument('--params', type=str, default='{}',
                        help='JSON of ClassicalBallTracker keyword arguments')
    parser.add_argument('--clips', type=str, default=None,
                        help='comma-separated game*/Clip* to evaluate instead of the whole val split')
    args = parser.parse_args()
    from hawk_eye.evaluation.ball import evaluate
    evaluate(args)


if __name__ == '__main__':
    main()
