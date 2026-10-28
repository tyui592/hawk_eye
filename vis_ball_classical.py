"""Save a video of the classical ball tracker output.

python vis_ball_classical.py --clip game10/Clip1 --save_dir ./data/outputs/classical_ball_v1_videos/
python vis_ball_classical.py --video_path ./data/DemoVideos/0.mp4 --save_dir ./data/outputs/classical_ball_v1_videos/

- --clip: a TrackNet clip (frames + Label.csv). Draws the ground truth and the per-frame
  status (TP/FN/FP/TN) with the same rule as `val_ball_classical.py`.
- --video_path: any video. Long videos are tracked in chunks of --chunk frames.

Drawing: red = tracked, yellow = interpolated inside a tracklet, fading trail of the last
--trail frames, green ring = ground truth, gray dots = candidates (--show_cands).
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from models.ball_detector.classical import ClassicalBallTracker
from val_ball_classical import load_labels, frame_status, SCORE_THRESHOLD

# BGR colors
COLOR_TRACKED = (0, 0, 255)
COLOR_INTERP = (0, 255, 255)
COLOR_GT = (0, 255, 0)
COLOR_CAND = (160, 160, 160)
COLOR_STATUS = {'tp': (0, 255, 0), 'fn': (0, 0, 255), 'fp': (0, 165, 255), 'tn': (200, 200, 200)}
MAX_TRAIL_STEP = 80  # px per frame


def open_writer(path, fps, size):
    """H.264 if available (plays in QuickTime), else MPEG-4."""
    for fourcc in ('avc1', 'mp4v'):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if writer.isOpened():
            return writer
    raise RuntimeError(f'cannot open a video writer for {path}')


def draw(img, t, positions, scores, cands=None, gt=None, status=None, trail=10, show_cands=False):
    """Draw the tracker output of frame t (and optionally ground truth / status) on a BGR image."""
    if show_cands and cands is not None:
        for c in cands:
            cv2.circle(img, (int(c[0]), int(c[1])), 2, COLOR_CAND, -1, cv2.LINE_AA)
    # fading trail
    for s in range(max(0, t - trail), t):
        if positions[s] is not None and positions[s + 1] is not None and scores[s] > SCORE_THRESHOLD:
            alpha = (s - t + trail + 1) / (trail + 1)
            color = tuple(int(v * alpha) for v in COLOR_TRACKED)
            p0 = tuple(map(round, positions[s]))
            p1 = tuple(map(round, positions[s + 1]))
            # do not connect two different tracklets
            if np.hypot(p1[0] - p0[0], p1[1] - p0[1]) < MAX_TRAIL_STEP:
                cv2.line(img, p0, p1, color, 2, cv2.LINE_AA)
    if gt is not None:
        cv2.circle(img, tuple(gt), 10, COLOR_GT, 1, cv2.LINE_AA)
    if positions[t] is not None and scores[t] > SCORE_THRESHOLD:
        color = COLOR_TRACKED if scores[t] == 1.0 else COLOR_INTERP
        cv2.circle(img, tuple(map(round, positions[t])), 5, color, 2, cv2.LINE_AA)
    text = f'{t:05d}'
    if status is not None:
        text += f'  {status.upper()}'
    cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                COLOR_STATUS.get(status, (255, 255, 255)), 2, cv2.LINE_AA)
    return img


def render_clip(clip, out_path, tracker, args):
    """Render a TrackNet clip with ground truth and per-frame status."""
    rows, gts = load_labels(clip)
    frames = [cv2.imread(str(clip / r['file name'])) for r in rows]
    positions, scores, cands = tracker(frames)
    h, w = frames[0].shape[:2]
    writer = open_writer(out_path, args.fps, (w, h))
    counts = {}
    for t, (img, gt) in enumerate(zip(frames, gts)):
        status, _ = frame_status(positions[t], scores[t], gt)
        counts[status] = counts.get(status, 0) + 1
        writer.write(draw(img, t, positions, scores, cands[t], gt, status, args.trail, args.show_cands))
    writer.release()
    return counts


def render_video(video_path, out_path, tracker, args):
    """Render any video, tracking it in chunks of `args.chunk` frames."""
    cap = cv2.VideoCapture(str(video_path))
    fps = args.fps or cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total = min(total, args.max_frames)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = open_writer(out_path, fps, (w, h))
    done, tracked = 0, 0
    with tqdm(total=total) as pbar:
        while done < total:
            frames = []
            while len(frames) < args.chunk and done + len(frames) < total:
                ok, img = cap.read()
                if not ok:
                    break
                frames.append(img)
            if not frames:
                break
            positions, scores, cands = tracker(frames)
            for t, img in enumerate(frames):
                tracked += scores[t] > SCORE_THRESHOLD
                writer.write(draw(img, t, positions, scores, cands[t], None, None, args.trail, args.show_cands))
            done += len(frames)
            pbar.update(len(frames))
    cap.release()
    writer.release()
    return {'frames': done, 'tracked': int(tracked)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=Path, default=Path('./data/TrackNet/'))
    parser.add_argument('--clip', type=str, default=None, help='comma-separated game*/Clip*')
    parser.add_argument('--video_path', type=Path, default=None)
    parser.add_argument('--save_dir', type=Path, default=Path('./data/outputs/classical_ball_v1_videos/'))
    parser.add_argument('--params', type=str, default='{}',
                        help='JSON of ClassicalBallTracker keyword arguments')
    parser.add_argument('--fps', type=float, default=None, help='default: 30 for clips, source fps for videos')
    parser.add_argument('--trail', type=int, default=10)
    parser.add_argument('--show_cands', action='store_true')
    parser.add_argument('--chunk', type=int, default=900)
    parser.add_argument('--max_frames', type=int, default=None)
    args = parser.parse_args()
    if (args.clip is None) == (args.video_path is None):
        parser.error('give exactly one of --clip or --video_path')

    tracker = ClassicalBallTracker(**json.loads(args.params))
    args.save_dir.mkdir(parents=True, exist_ok=True)
    if args.clip:
        args.fps = args.fps or 30
        for c in args.clip.split(','):
            out_path = args.save_dir / f"{c.replace('/', '_')}.mp4"
            counts = render_clip(args.data_path / c, out_path, tracker, args)
            print(out_path, counts)
    else:
        out_path = args.save_dir / f'{args.video_path.stem}.mp4'
        print(out_path, render_video(args.video_path, out_path, tracker, args))


if __name__ == '__main__':
    main()
