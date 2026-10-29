"""Validation Code for the classical court detector.

Uses the same metric as `hawk_eye/engine/validate.py` (score > 0.5 and distance < 7 pixels).

python -m hawk_eye.cli.eval_court --data_path ./data/TennisCourtDetector/ --num_workers 8 \
    --save_dir ./data/outputs/court_detection/classical-v3/tenniscourt-val-new/

With --save_dir, saves
- results.json: raw outputs (keypoints, scores, homography, ...) and per-point status
- summary.csv: one row per image sorted by the number of wrong points
- fail_triage.csv: failed rows grouped by mismatch extent
- metrics.txt: overall metrics
- vis/fail/*.jpg, vis/ok/*.jpg: visualizations selected by --save_vis
"""

import csv
import json
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

from hawk_eye.models.court_detector.classical import ClassicalCourtDetector, MODEL_SEGMENTS, project
from hawk_eye.models.court_detector.loss import Criterion
from hawk_eye.experiments.outputs import start_run, finish_run, model_parameters

SCORE_THRESHOLD = 0.5
DISTANCE_THRESHOLD = 7

# BGR colors
COLOR_COURT = (0, 255, 255)
COLOR_GT = (255, 255, 255)
COLOR_STATUS = {'tp': (0, 255, 0), 'fn': (0, 0, 255), 'fp': (0, 165, 255)}


def point_status(keypoint, score, gt):
    """Status of a keypoint with the same rule as `Criterion.calc_confusion`.

    gt: (x, y) or None when the keypoint is not visible.
    """
    if gt is not None:
        dist = np.hypot(keypoint[0] - gt[0], keypoint[1] - gt[1])
        if score > SCORE_THRESHOLD and dist < DISTANCE_THRESHOLD:
            return 'tp', dist
        return 'fn', dist
    if score > SCORE_THRESHOLD:
        return 'fp', None
    return 'tn', None


def visualize(img, result):
    """Draw the projected court, predictions and ground truths on a BGR image."""
    if result['homography'] is not None:
        H = np.array(result['homography'])
        for segment in MODEL_SEGMENTS:
            (p0, p1), _ = project(H, np.array(segment, dtype=np.float64))
            cv2.line(img, tuple(map(round, p0)), tuple(map(round, p1)), COLOR_COURT, 1, cv2.LINE_AA)

    for i, (kp, gt, status) in enumerate(zip(result['keypoints'], result['gt_keypoints'], result['status'])):
        if gt is not None:
            cv2.circle(img, tuple(map(round, gt)), DISTANCE_THRESHOLD, COLOR_GT, 1, cv2.LINE_AA)
        if status == 'tn':
            continue
        x, y = map(round, kp)
        color = COLOR_STATUS[status]
        if gt is not None and status == 'fn':
            cv2.line(img, (x, y), tuple(map(round, gt)), color, 1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 3, color, -1, cv2.LINE_AA)
        cv2.putText(img, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    header = (f"{result['id']}  wrong={result['num_wrong']}  "
              f"court_score={result['court_score']:.3f}  mean_err={result['mean_error']:.2f}px")
    legend = 'green: TP  red: FN (line to GT)  orange: FP  white circle: GT (7px)  yellow: court'
    for text, y in [(header, 28), (legend, 52)]:
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def process(item):
    """Run the detector on a single image and optionally save the visualization."""
    image_path, gt_kps, split, save_dir, save_vis = item
    bgr = cv2.imread(str(image_path))
    h, w = bgr.shape[:2]
    detector = ClassicalCourtDetector()
    keypoints, scores = detector(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    # same visibility rule as hawk_eye/datasets/court.py
    gt_visible = [(x, y) if 0 <= x < w and 0 <= y < h else None for x, y in gt_kps]
    status, errors = zip(*[point_status(kp, sc, gt) for kp, sc, gt in zip(keypoints, scores, gt_visible)])
    valid_errors = [e for e in errors if e is not None]

    result = {
        'id': image_path.stem,
        'split': split,
        'image_size': [h, w],
        'keypoints': keypoints,
        'scores': scores,
        'gt_keypoints': gt_visible,
        'status': list(status),
        'errors': list(errors),
        'num_wrong': sum(s in ('fn', 'fp') for s in status),
        'mean_error': float(np.mean(valid_errors)) if valid_errors else 0.0,
        'homography': detector.matrix.tolist() if detector.matrix is not None else None,
        'court_score': float(detector.score),
    }

    should_save = save_dir is not None and (save_vis == 'all' or
                                             (save_vis == 'fail' and result['num_wrong'] > 0))
    if should_save:
        sub = 'fail' if result['num_wrong'] > 0 else 'ok'
        result['vis'] = f"vis/{sub}/{result['id']}.jpg"
        cv2.imwrite(str(save_dir / 'vis' / sub / f"{result['id']}.jpg"), visualize(bgr, result),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    else:
        result['vis'] = None
    return result


def save_fail_triage(save_dir, results):
    """Save failed rows grouped by the extent of their mismatch."""
    with open(save_dir / 'fail_triage.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'id', 'kind', 'num_wrong', 'court_score', 'mean_error', 'vis'])
        for r in sorted(results, key=lambda r: (-r['num_wrong'], r['court_score'])):
            if r['num_wrong'] == 0:
                continue
            if r['num_wrong'] == 1:
                kind = 'single_point'
            elif r['num_wrong'] <= 3:
                kind = 'partial_2_3'
            else:
                kind = 'global_4_plus'
            writer.writerow([r['split'], r['id'], kind, r['num_wrong'],
                             f"{r['court_score']:.4f}", f"{r['mean_error']:.2f}", r['vis'] or ''])


def save_results(save_dir, results, metrics):
    """Save raw results, per-image summaries and metrics."""
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(results, f)

    with open(save_dir / 'summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'id', 'num_wrong', 'wrong_points', 'court_score', 'mean_error', 'vis'])
        for r in sorted(results, key=lambda r: (-r['num_wrong'], r['court_score'])):
            wrong = [str(i) for i, s in enumerate(r['status']) if s in ('fn', 'fp')]
            writer.writerow([r['split'], r['id'], r['num_wrong'], ' '.join(wrong),
                             f"{r['court_score']:.4f}", f"{r['mean_error']:.2f}", r['vis'] or ''])

    with open(save_dir / 'metrics.txt', 'w') as f:
        f.write(metrics + '\n')
    save_fail_triage(save_dir, results)


def calculate_metrics(results):
    """Calculate the existing court metric for a list of result rows."""
    criterion = Criterion(alpha=2.0, beta=4.0)
    confusion = defaultdict(int)
    errors = []
    for result in results:
        kps = [gt for gt in result['gt_keypoints'] if gt is not None]
        labels = [str(i) for i, gt in enumerate(result['gt_keypoints']) if gt is not None]
        targets = {'raw_keypoints': [kps], 'raw_labels': [labels]}
        criterion.calc_confusion(
            confusion, [result['keypoints']], [result['scores']], targets,
            score_threshold=SCORE_THRESHOLD, distance_threshold=DISTANCE_THRESHOLD,
        )
        errors.append(criterion.calc_pixel_dist([result['keypoints']], [result['scores']], targets))
    recall = confusion['tp'] / max(confusion['tp'] + confusion['fn'], 1)
    precision = confusion['tp'] / max(confusion['tp'] + confusion['fp'], 1)
    return confusion, float(np.mean(errors)), recall, precision


def evaluate(args):

    if args.save_dir is not None:
        args.save_dir = start_run(args.save_dir, 'court_detection', 'classical-v3', 'evaluation',
                                  [str(args.data_path / f'data_{split}.json')
                                   for split in (['train', 'val'] if args.image_set == 'all' else [args.image_set])],
                                  {**model_parameters(ClassicalCourtDetector), 'save_vis': args.save_vis})
        if args.save_vis in ('all', 'fail'):
            (args.save_dir / 'vis' / 'fail').mkdir(parents=True, exist_ok=True)
        if args.save_vis == 'all':
            (args.save_dir / 'vis' / 'ok').mkdir(parents=True, exist_ok=True)

    splits = ['train', 'val'] if args.image_set == 'all' else [args.image_set]
    labels = [(split, data) for split in splits
              for data in json.load(open(args.data_path / f'data_{split}.json'))]
    items = [(args.data_path / 'images' / f"{data['id']}.png", data['kps'], split,
              args.save_dir, args.save_vis) for split, data in labels]

    with ProcessPoolExecutor(args.num_workers) as executor:
        results = list(tqdm(executor.map(process, items, chunksize=4), total=len(items)))

    confusion, error, recall, precision = calculate_metrics(results)
    num_fail = sum(r['num_wrong'] > 0 for r in results)
    metrics = '\n'.join([
        f"Images: {len(results)}, Images with wrong points: {num_fail}",
        f"Error: {error:1.4f}",
        f"TP: {confusion['tp']}, FP: {confusion['fp']}, TN: {confusion['tn']}, FN: {confusion['fn']}",
        f"Recall: {recall:1.4f}, Precision: {precision:1.4f}",
    ])
    split_summaries = {}
    if args.image_set == 'all':
        for split in splits:
            subset = [r for r in results if r['split'] == split]
            split_confusion, split_error, split_recall, split_precision = calculate_metrics(subset)
            split_wrong = sum(r['num_wrong'] > 0 for r in subset)
            split_summaries[split] = {
                'images': len(subset), 'wrong_images': split_wrong, 'error': split_error,
                'recall': split_recall, 'precision': split_precision,
                'confusion': dict(split_confusion),
            }
            metrics += (f'\n{split}: Images {len(subset)}, wrong {split_wrong}, '
                        f'Error {split_error:.4f}, Recall {split_recall:.4f}, '
                        f'Precision {split_precision:.4f}')
    print(metrics)

    if args.save_dir is not None:
        save_results(args.save_dir, results, metrics)
        summary = {
            'images': len(results), 'wrong_images': num_fail, 'error': error,
            'recall': recall, 'precision': precision, 'confusion': dict(confusion),
        }
        if split_summaries:
            summary['splits'] = split_summaries
        finish_run(args.save_dir, summary)
        print(f"Saved to {args.save_dir}")
