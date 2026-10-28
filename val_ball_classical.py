"""Validation Code for the classical ball tracker.

Uses the same metric as `val.py` (`models/ball_detector/loss.py` `calc_confusion`):
score > 0.5 and distance < 7 pixels in the raw 1280x720 frame. Positives are
visibility '1' and '2' with integer coordinates, as in `datasets/ball.py`.

Reads the raw TrackNet layout (game*/Clip*/Label.csv). The val split is game9 and
game10 (the original data_val.pkl split is unknown).

python val_ball_classical.py --data_path ./data/TrackNet/ --num_workers 8 \
    --save_dir ./data/outputs/classical_ball_val/

With --save_dir, saves
- results.json: per-frame prediction, score, ground truth and status
- summary.csv: one row per clip
- metrics.txt: overall metrics
"""

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

from models.ball_detector.classical import ClassicalBallTracker

SCORE_THRESHOLD = 0.5
DISTANCE_THRESHOLD = 7
FOREGROUNDS = ('1', '2')
VAL_GAMES = ('game9', 'game10')


def frame_status(position, score, gt):
    """Status of a frame with the same rule as `Criterion.calc_confusion`."""
    if gt is not None:
        dist = np.hypot(position[0] - gt[0], position[1] - gt[1]) if position is not None else np.inf
        if score > SCORE_THRESHOLD and dist < DISTANCE_THRESHOLD:
            return 'tp', dist
        return 'fn', dist
    if score > SCORE_THRESHOLD:
        return 'fp', None
    return 'tn', None


def load_labels(clip):
    """Rows of Label.csv and ground truths ((x, y) or None)."""
    with open(clip / 'Label.csv') as f:
        rows = list(csv.DictReader(f))
    gts = []
    for r in rows:
        gt = None
        if r['visibility'] in FOREGROUNDS:
            x, y = int(float(r['x-coordinate'])), int(float(r['y-coordinate']))
            if 0 <= x < 1280 and 0 <= y < 720:
                gt = (x, y)
        gts.append(gt)
    return rows, gts


def run_clip(args):
    """Track one clip and compute per-frame statuses."""
    clip, params = args
    rows, gts = load_labels(clip)
    # Read BGR like a video stream; IMREAD_GRAYSCALE decodes JPEG luma differently around the
    # (small, yellow) ball and changed game10/Clip1 TP from 305 to 262.
    frames = [cv2.imread(str(clip / r['file name'])) for r in rows]
    tracker = ClassicalBallTracker(**params)
    positions, scores, cands = tracker(frames)

    results = []
    for r, gt, pos, score, cs in zip(rows, gts, positions, scores, cands):
        status, dist = frame_status(pos, score, gt)
        cand_hit = None
        if gt is not None:
            cand_hit = bool(len(cs)) and bool(np.min(np.hypot(cs[:, 0] - gt[0], cs[:, 1] - gt[1])) < DISTANCE_THRESHOLD)
        results.append({
            'file_name': f"{clip.parent.name}/{clip.name}/{r['file name']}",
            'visibility': r['visibility'],
            'status_label': r['status'],
            'gt': gt,
            'pred': None if pos is None else [float(pos[0]), float(pos[1])],
            'score': float(score),
            'status': status,
            'dist': None if dist is None or not np.isfinite(dist) else float(dist),
            'num_cands': int(len(cs)),
            'cand_hit': cand_hit,
        })
    return results


def summarize(results):
    """Confusion, recall, precision and candidate recall."""
    confusion = defaultdict(int)
    for r in results:
        confusion[r['status']] += 1
    tp, fp, fn = confusion['tp'], confusion['fp'], confusion['fn']
    pos = [r for r in results if r['gt'] is not None]
    return {
        'tp': tp, 'fp': fp, 'tn': confusion['tn'], 'fn': fn,
        'recall': tp / max(tp + fn, 1),
        'precision': tp / max(tp + fp, 1),
        'cand_recall': sum(r['cand_hit'] for r in pos) / max(len(pos), 1),
    }


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

    params = json.loads(args.params)
    if args.clips:
        clips = [args.data_path / c for c in args.clips.split(',')]
    else:
        clips = [c for g in VAL_GAMES for c in sorted((args.data_path / g).glob('Clip*'), key=lambda p: int(p.name[4:]))]
    with ProcessPoolExecutor(args.num_workers) as ex:
        per_clip = list(tqdm(ex.map(run_clip, [(c, params) for c in clips]), total=len(clips)))
    results = [r for rs in per_clip for r in rs]

    m = summarize(results)
    lines = [
        f"Clips: {len(clips)}, Frames: {len(results)}, Params: {json.dumps(params)}",
        f"TP: {m['tp']}, FP: {m['fp']}, TN: {m['tn']}, FN: {m['fn']}",
        f"Recall: {m['recall']:1.4f}, Precision: {m['precision']:1.4f}",
        f"Candidate recall (GT within 7px of any candidate): {m['cand_recall']:1.4f}",
    ]
    for vis in FOREGROUNDS:
        sub = summarize([r for r in results if r['visibility'] == vis])
        lines.append(f"  visibility {vis}: recall {sub['recall']:1.4f}, cand recall {sub['cand_recall']:1.4f}")
    print('\n'.join(lines))

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=False)
        (args.save_dir / 'metrics.txt').write_text('\n'.join(lines) + '\n')
        with open(args.save_dir / 'results.json', 'w') as f:
            json.dump(results, f)
        with open(args.save_dir / 'summary.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['clip', 'frames', 'tp', 'fp', 'tn', 'fn', 'recall', 'precision', 'cand_recall'])
            for clip, rs in zip(clips, per_clip):
                s = summarize(rs)
                writer.writerow([f'{clip.parent.name}/{clip.name}', len(rs), s['tp'], s['fp'], s['tn'], s['fn'],
                                 f"{s['recall']:.4f}", f"{s['precision']:.4f}", f"{s['cand_recall']:.4f}"])


if __name__ == '__main__':
    main()
