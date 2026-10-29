"""Validation Code for the classical ball tracker.

Uses the same metric as `hawk_eye/engine/validate.py` (`hawk_eye/models/ball_detector/loss.py` `calc_confusion`):
score > 0.5 and distance < 7 pixels in the raw 1280x720 frame. Positives are
visibility '1' and '2' with integer coordinates, as in `hawk_eye/datasets/ball.py`.

Reads the raw TrackNet layout (game*/Clip*/Label.csv). The val split is game9 and
game10 (the original data_val.pkl split is unknown).

python -m hawk_eye.cli.eval_ball --data_path ./data/TrackNet/ --num_workers 8 \
    --save_dir ./data/outputs/ball_tracking/classical-v1b/tracknet-game9-game10-new/

With --save_dir, saves
- results.json: per-frame prediction, score, ground truth and status
- summary.csv: one row per clip
- metrics.txt: overall metrics
"""

import csv
import json
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

from hawk_eye.models.ball_detector.classical import ClassicalBallTracker
from hawk_eye.experiments.outputs import start_run, finish_run, model_parameters

from hawk_eye.evaluation.tracknet import (
    SCORE_THRESHOLD, DISTANCE_THRESHOLD, FOREGROUNDS, frame_status, load_labels,
)

VAL_GAMES = ('game9', 'game10')


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


def evaluate(args):

    params = json.loads(args.params)
    if args.clips:
        clips = [args.data_path / c for c in args.clips.split(',')]
    else:
        clips = [c for g in VAL_GAMES for c in sorted((args.data_path / g).glob('Clip*'), key=lambda p: int(p.name[4:]))]
    if args.save_dir is not None:
        version = 'classical-v3b-motion' if params.get('motion_prior', False) else 'classical-v1b'
        args.save_dir = start_run(args.save_dir, 'ball_tracking', version, 'evaluation',
                                  [str(c) for c in clips], model_parameters(ClassicalBallTracker, params))
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
        finish_run(args.save_dir, m)
