"""Save a video of the classical ball tracker output.

python vis_ball_classical.py --clip game10/Clip1
python vis_ball_classical.py --video_path ./data/DemoVideos/0.mp4

- --clip: a TrackNet clip (frames + Label.csv). Draws the ground truth and the per-frame
  status (TP/FN/FP/TN) with the same rule as `val_ball_classical.py`.
- --video_path: any video. Long videos are tracked in chunks of --chunk frames.
- Court gating is enabled by default; --no_court_gate disables it for comparison.

Drawing: red = tracked, yellow = interpolated or repaired, fading trail of the last
--trail frames, green ring = ground truth, gray dots = candidates (--show_cands).
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from models.ball_detector.classical import ClassicalBallTracker
from models.ball_detector.court_gate import CourtGate, track_court_segments
from models.court_detector.classical import ClassicalCourtDetector
from utils.experiment_outputs import start_run, finish_run, model_parameters
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


def draw(img, t, positions, scores, cands=None, gt=None, status=None, trail=10, show_cands=False, index=None,
         court_records=None):
    """Draw the tracker output of frame t (and optionally ground truth / status) on a BGR image.

    index: frame number to print (default t).
    """
    if show_cands and cands is not None:
        for c in cands:
            cv2.circle(img, (int(c[0]), int(c[1])), 2, COLOR_CAND, -1, cv2.LINE_AA)
    # fading trail
    trail_start = max(0, t - trail)
    if court_records is not None:
        if not court_records[t]['active']:
            trail_start = t
        else:
            for s in range(t - 1, trail_start - 1, -1):
                if not court_records[s]['active'] or court_records[s]['shot'] != court_records[t]['shot']:
                    trail_start = s + 1
                    break
    for s in range(trail_start, t):
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
    text = f'{t if index is None else index:05d}'
    if status is not None:
        text += f'  {status.upper()}'
    cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                COLOR_STATUS.get(status, (255, 255, 255)), 2, cv2.LINE_AA)
    if court_records is not None:
        record = court_records[t]
        label = f"COURT {'ON' if record['active'] else 'OFF'}  score={record['score']:.2f}"
        color = COLOR_GT if record['active'] else (0, 165, 255)
        cv2.putText(img, label, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, label, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return img


def make_gate(args, fps):
    """Create an optional court gate with a one-second default check interval."""
    if args.no_court_gate:
        return None
    return CourtGate(interval=args.court_interval or max(1, round(fps)), min_score=args.court_min_score)


def render_clip(clip, out_path, tracker, args):
    """Render a TrackNet clip with ground truth and per-frame status."""
    rows, gts = load_labels(clip)
    frames = [cv2.imread(str(clip / r['file name'])) for r in rows]
    tracker.fps = args.fps
    gate = make_gate(args, args.fps)
    records = gate(frames) if gate else None
    positions, scores, cands = track_court_segments(frames, tracker, records) if gate else tracker(frames)
    h, w = frames[0].shape[:2]
    writer = open_writer(out_path, args.fps, (w, h))
    counts = {}
    for t, (img, gt) in enumerate(zip(frames, gts)):
        status, _ = frame_status(positions[t], scores[t], gt)
        counts[status] = counts.get(status, 0) + 1
        writer.write(draw(img, t, positions, scores, cands[t], gt, status, args.trail, args.show_cands,
                          court_records=records))
    writer.release()
    return counts


def render_video(video_path, out_path, tracker, args):
    """Render any video, tracking it in chunks of `args.chunk` frames."""
    cap = cv2.VideoCapture(str(video_path))
    fps = args.fps or cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        raise ValueError(f'cannot read video: {video_path}')
    gate = make_gate(args, fps)
    tracker.fps = fps
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames:
        total = min(total, args.max_frames)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = open_writer(out_path, fps, (w, h))
    comparison = open_writer(out_path.with_name(out_path.stem + '_comparison.mp4'), fps, (w * 2, h)) \
        if args.compare_motion else None
    done, tracked = 0, 0
    court_frames, records_all = 0, []
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
            records = gate(frames) if gate else None
            baseline = None
            if comparison is not None:
                enabled = tracker.motion_prior
                tracker.motion_prior = False
                try:
                    baseline = track_court_segments(frames, tracker, records) if gate else tracker(frames)
                finally:
                    tracker.motion_prior = enabled
            positions, scores, cands = track_court_segments(frames, tracker, records) if gate else tracker(frames)
            if records is not None:
                court_frames += sum(r['active'] for r in records)
                records_all.extend(dict(r, frame=done + t, tracked=bool(scores[t] > SCORE_THRESHOLD))
                                   for t, r in enumerate(records))
                for t, r in enumerate(records_all[-len(frames):]):
                    r['pred'] = None if positions[t] is None else list(map(float, positions[t]))
                    if baseline is not None:
                        r['baseline_pred'] = None if baseline[0][t] is None else list(map(float, baseline[0][t]))
            for t, img in enumerate(frames):
                tracked += scores[t] > SCORE_THRESHOLD
                before = None
                if comparison is not None:
                    before = draw(img.copy(), t, baseline[0], baseline[1], baseline[2][t],
                                  trail=args.trail, index=done + t, court_records=records)
                    cv2.putText(before, 'BEFORE', (20, h - 20), 0, 1, (255, 255, 255), 2)
                after = draw(img, t, positions, scores, cands[t], None, None, args.trail, args.show_cands,
                             index=done + t, court_records=records)
                writer.write(after)
                if comparison is not None:
                    cv2.putText(after, 'MOTION PRIOR', (20, h - 20), 0, 1, (255, 255, 255), 2)
                    comparison.write(np.hstack([before, after]))
            done += len(frames)
            pbar.update(len(frames))
    cap.release()
    writer.release()
    if comparison is not None:
        comparison.release()
    summary = {'frames': done, 'tracked': int(tracked), 'court_gate': gate is not None,
               'fps': fps, 'width': w, 'height': h}
    summary['tracker_params'] = json.loads(args.params)
    if gate:
        summary.update(court_frames=int(court_frames), court_checks=gate.checks)
        out_path.with_suffix('.court.json').write_text(json.dumps(records_all), encoding='utf-8')
    out_path.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


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

    params = json.loads(args.params)
    tracker = ClassicalBallTracker(**params)
    target = 'ball_tracking' if args.no_court_gate else 'pipeline'
    version = ('classical-v3b-motion' if tracker.motion_prior else 'classical-v1b') if args.no_court_gate else \
        ('classical-v3b-court-gate-motion' if tracker.motion_prior else 'classical-v2-court-gate')
    settings = dict(ball=model_parameters(ClassicalBallTracker, params),
                    court=model_parameters(ClassicalCourtDetector) if not args.no_court_gate else None,
                    court_gate=not args.no_court_gate, court_interval=args.court_interval,
                    court_min_score=args.court_min_score, fps_override=args.fps,
                    chunk=args.chunk, max_frames=args.max_frames, compare_motion=args.compare_motion)
    args.save_dir = start_run(args.save_dir, target, version, 'demo',
                              args.clip or str(args.video_path), settings)
    summaries = {}
    if args.clip:
        args.fps = args.fps or 30
        for c in args.clip.split(','):
            out_path = args.save_dir / f"{c.replace('/', '_')}.mp4"
            counts = render_clip(args.data_path / c, out_path, tracker, args)
            summaries[c] = counts
            print(out_path, counts)
    else:
        out_path = args.save_dir / f'{args.video_path.stem}.mp4'
        summaries = render_video(args.video_path, out_path, tracker, args)
        print(out_path, summaries)
    finish_run(args.save_dir, summaries)


if __name__ == '__main__':
    main()
