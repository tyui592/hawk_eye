"""Validation Code for the classical court detector.

Uses the same metric as `val.py` (score > 0.5 and distance < 7 pixels).

python val_classical.py --data_path ./data/TennisCourtDetector/ --num_workers 8 \
    --save_dir ./data/outputs/court_detection/classical-v3/tenniscourt-val-new/

With --save_dir, saves
- results.json: raw outputs (keypoints, scores, homography, ...) and per-point status
- summary.csv: one row per image sorted by the number of wrong points
- metrics.txt: overall metrics
- vis/fail/*.jpg, vis/ok/*.jpg: visualizations
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

from models.court_detector.classical import ClassicalCourtDetector, MODEL_SEGMENTS, project
from models.court_detector.loss import Criterion
from utils.experiment_outputs import start_run, finish_run, model_parameters

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
    image_path, gt_kps, save_dir = item
    bgr = cv2.imread(str(image_path))
    h, w = bgr.shape[:2]
    detector = ClassicalCourtDetector()
    keypoints, scores = detector(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    # same visibility rule as datasets/court.py
    gt_visible = [(x, y) if 0 <= x < w and 0 <= y < h else None for x, y in gt_kps]
    status, errors = zip(*[point_status(kp, sc, gt) for kp, sc, gt in zip(keypoints, scores, gt_visible)])
    valid_errors = [e for e in errors if e is not None]

    result = {
        'id': image_path.stem,
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

    if save_dir is not None:
        sub = 'fail' if result['num_wrong'] > 0 else 'ok'
        cv2.imwrite(str(save_dir / 'vis' / sub / f"{result['id']}.jpg"), visualize(bgr, result),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    return result


def save_results(save_dir, results, metrics):
    """Save raw results, a per-image summary and metrics."""
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(results, f)

    with open(save_dir / 'summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'num_wrong', 'wrong_points', 'court_score', 'mean_error', 'vis'])
        for r in sorted(results, key=lambda r: (-r['num_wrong'], r['court_score'])):
            wrong = [str(i) for i, s in enumerate(r['status']) if s in ('fn', 'fp')]
            sub = 'fail' if r['num_wrong'] > 0 else 'ok'
            writer.writerow([r['id'], r['num_wrong'], ' '.join(wrong), f"{r['court_score']:.4f}",
                             f"{r['mean_error']:.2f}", f"vis/{sub}/{r['id']}.jpg"])

    with open(save_dir / 'metrics.txt', 'w') as f:
        f.write(metrics + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=Path, default=Path('./data/TennisCourtDetector/'))
    parser.add_argument('--image_set', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_dir', type=Path, default=None,
                        help='Directory to save raw results and visualizations')
    args = parser.parse_args()

    if args.save_dir is not None:
        args.save_dir = start_run(args.save_dir, 'court_detection', 'classical-v3', 'evaluation',
                                  str(args.data_path / f'data_{args.image_set}.json'),
                                  model_parameters(ClassicalCourtDetector))
        (args.save_dir / 'vis' / 'ok').mkdir(parents=True, exist_ok=True)
        (args.save_dir / 'vis' / 'fail').mkdir(parents=True, exist_ok=True)

    labels = json.load(open(args.data_path / f'data_{args.image_set}.json'))
    items = [(args.data_path / 'images' / f"{data['id']}.png", data['kps'], args.save_dir) for data in labels]

    with ProcessPoolExecutor(args.num_workers) as executor:
        results = list(tqdm(executor.map(process, items, chunksize=4), total=len(items)))

    criterion = Criterion(alpha=2.0, beta=4.0)
    confusion = defaultdict(int)
    errors = []
    for r in results:
        kps = [gt for gt in r['gt_keypoints'] if gt is not None]
        kp_labels = [str(i) for i, gt in enumerate(r['gt_keypoints']) if gt is not None]
        targets = {'raw_keypoints': [kps], 'raw_labels': [kp_labels]}
        criterion.calc_confusion(confusion, [r['keypoints']], [r['scores']], targets)
        errors.append(criterion.calc_pixel_dist([r['keypoints']], [r['scores']], targets))

    recall = confusion['tp'] / (confusion['tp'] + confusion['fn'])
    precision = confusion['tp'] / (confusion['tp'] + confusion['fp'])
    num_fail = sum(r['num_wrong'] > 0 for r in results)
    metrics = '\n'.join([
        f"Images: {len(results)}, Images with wrong points: {num_fail}",
        f"Error: {np.mean(errors):1.4f}",
        f"TP: {confusion['tp']}, FP: {confusion['fp']}, TN: {confusion['tn']}, FN: {confusion['fn']}",
        f"Recall: {recall:1.4f}, Precision: {precision:1.4f}",
    ])
    print(metrics)

    if args.save_dir is not None:
        save_results(args.save_dir, results, metrics)
        finish_run(args.save_dir, dict(recall=recall, precision=precision, images=len(results)))
        print(f"Saved to {args.save_dir}")


if __name__ == '__main__':
    main()
