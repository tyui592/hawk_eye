"""Run visualize ball from the command line."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=Path, default=Path('./data/TrackNet/'))
    parser.add_argument('--clip', type=str, default=None, help='comma-separated game*/Clip*')
    parser.add_argument('--video_path', type=Path, default=None)
    parser.add_argument('--save_dir', type=Path, default=None,
                        help='fresh output directory; default: target/version/timestamp under data/outputs')
    parser.add_argument('--params', type=str, default='{}',
                        help='JSON of ClassicalBallTracker keyword arguments')
    parser.add_argument('--fps', type=float, default=None, help='default: 30 for clips, source fps for videos')
    parser.add_argument('--trail', type=int, default=10)
    parser.add_argument('--show_cands', action='store_true')
    parser.add_argument('--chunk', type=int, default=900)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--no_court_gate', action='store_true', help='disable court gating for comparison')
    parser.add_argument('--court_interval', type=int, default=None, help='court check interval in frames (default: 1 second)')
    parser.add_argument('--court_min_score', type=float, default=0.5)
    parser.add_argument('--compare_motion', action='store_true', help='also save a side-by-side prior-off/on video')
    args = parser.parse_args()
    if (args.clip is None) == (args.video_path is None):
        parser.error('give exactly one of --clip or --video_path')
    if args.chunk < 3 or (args.court_interval is not None and args.court_interval < 1):
        parser.error('chunk must be >= 3 and court_interval must be positive')
    if args.compare_motion and (args.video_path is None or not json.loads(args.params).get('motion_prior')):
        parser.error('compare_motion requires video_path and motion_prior=true')

    from hawk_eye.evaluation.visualize_ball import run
    run(args)


if __name__ == '__main__':
    main()
