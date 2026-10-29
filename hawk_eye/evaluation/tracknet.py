"""Read TrackNet labels and classify frame predictions."""

import csv
import numpy as np

SCORE_THRESHOLD = 0.5
DISTANCE_THRESHOLD = 7
FOREGROUNDS = ('1', '2')


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
