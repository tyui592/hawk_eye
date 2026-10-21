"""Validation Code for the classical court detector.

Uses the same metric as `val.py` (score > 0.5 and distance < 7 pixels).

python val_classical.py --data_path ./data/TennisCourtDetector/ --num_workers 8
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

from models.court_detector.classical import ClassicalCourtDetector
from models.court_detector.loss import Criterion


def detect(image_path):
    """Run the detector on a single image."""
    img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    detector = ClassicalCourtDetector()
    keypoints, scores = detector(img)
    return keypoints, scores, img.shape[:2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=Path, default=Path('./data/TennisCourtDetector/'))
    parser.add_argument('--image_set', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    labels = json.load(open(args.data_path / f'data_{args.image_set}.json'))
    image_paths = [args.data_path / 'images' / f"{data['id']}.png" for data in labels]

    with ProcessPoolExecutor(args.num_workers) as executor:
        results = list(tqdm(executor.map(detect, image_paths, chunksize=4), total=len(image_paths)))

    criterion = Criterion(alpha=2.0, beta=4.0)
    confusion = defaultdict(int)
    errors = []
    for data, (keypoints, scores, (h, w)) in zip(labels, results):
        # same visibility rule as datasets/court.py
        kps, kp_labels = [], []
        for i, (x, y) in enumerate(data['kps']):
            if 0 <= x < w and 0 <= y < h:
                kps.append((x, y))
                kp_labels.append(str(i))
        targets = {'raw_keypoints': [kps], 'raw_labels': [kp_labels]}
        criterion.calc_confusion(confusion, [keypoints], [scores], targets)
        errors.append(criterion.calc_pixel_dist([keypoints], [scores], targets))

    recall = confusion['tp'] / (confusion['tp'] + confusion['fn'])
    precision = confusion['tp'] / (confusion['tp'] + confusion['fp'])
    print(f"Error: {np.mean(errors):1.4f}")
    print(f"TP: {confusion['tp']}, FP: {confusion['fp']}, TN: {confusion['tn']}, FN: {confusion['fn']}")
    print(f"Recall: {recall:1.4f}, Precision: {precision:1.4f}")


if __name__ == '__main__':
    main()
