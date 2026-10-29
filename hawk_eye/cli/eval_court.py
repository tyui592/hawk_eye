"""Run court from the command line."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=Path, default=Path('./data/TennisCourtDetector/'))
    parser.add_argument('--image_set', type=str, default='val', choices=['train', 'val', 'all'])
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_dir', type=Path, default=None,
                        help='Directory to save raw results and visualizations')
    parser.add_argument('--save_vis', choices=['all', 'fail', 'none'], default='all',
                        help='visualizations to save when save_dir is set')
    args = parser.parse_args()
    from hawk_eye.evaluation.court import evaluate
    evaluate(args)


if __name__ == '__main__':
    main()
