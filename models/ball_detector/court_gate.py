"""Run ball tracking only in shots with a detected court."""

import cv2
import numpy as np

from models.court_detector.classical import ClassicalCourtDetector


class CourtGate:
    """Check court evidence periodically and immediately after a shot change."""

    def __init__(self, interval=25, min_score=0.5, cut_threshold=25.0):
        if interval < 1:
            raise ValueError('court interval must be positive')
        self.interval = interval
        self.min_score = min_score
        self.cut_threshold = cut_threshold
        self.detector = ClassicalCourtDetector()
        self.previous = None
        self.remaining = 0
        self.shot = 0
        self.active = False
        self.score = 0.0
        self.checks = 0

    def __call__(self, frames):
        """Return per-frame gate state, court score, and shot id for BGR frames."""
        records = []
        for frame in frames:
            thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (96, 54))
            cut = self.previous is not None and float(np.mean(cv2.absdiff(thumb, self.previous))) > self.cut_threshold
            if cut:
                self.shot += 1
            checked = self.previous is None or cut or self.remaining <= 0
            if checked:
                _, visible = self.detector(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                self.score = float(self.detector.score)
                # Keypoint scores describe visibility, not court confidence.
                self.active = (self.detector.matrix is not None and np.isfinite(self.score)
                               and self.score >= self.min_score and sum(visible) >= 4)
                self.remaining = self.interval
                self.checks += 1
            records.append({'active': bool(self.active), 'score': self.score,
                            'shot': self.shot, 'checked': checked, 'cut': bool(cut)})
            self.previous = thumb
            self.remaining -= 1
        return records


def track_court_segments(frames, tracker, records):
    """Track each accepted run separately, without bridging disabled frames or cuts."""
    if len(frames) != len(records):
        raise ValueError('court records must match frames')
    positions = [None] * len(frames)
    scores = [0.0] * len(frames)
    candidates = [np.zeros((0, 5)) for _ in frames]
    start = 0
    while start < len(frames):
        if not records[start]['active']:
            start += 1
            continue
        end = start + 1
        while (end < len(frames) and records[end]['active']
               and records[end]['shot'] == records[start]['shot']):
            end += 1
        if end - start >= 3:
            p, s, c = tracker(frames[start:end])
            positions[start:end], scores[start:end], candidates[start:end] = p, s, c
        start = end
    return positions, scores, candidates
