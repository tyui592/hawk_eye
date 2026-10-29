"""Line Operations.

A line is represented as [x1, y1, x2, y2] (output format of cv2.HoughLinesP).
"""

import numpy as np


def line_angle(line):
    """Angle of a line in degree, [0, 180)."""
    x1, y1, x2, y2 = line
    return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180


def point_line_distance(point, line):
    """Perpendicular distance from a point to an (infinite) line."""
    x1, y1, x2, y2 = line
    px, py = point
    norm = np.hypot(x2 - x1, y2 - y1)
    if norm == 0:
        return np.hypot(px - x1, py - y1)
    return abs((x2 - x1) * (y1 - py) - (x1 - px) * (y2 - y1)) / norm


def cluster_lines(lines, angle_threshold=10.0, distance_threshold=5.0):
    """Greedily group lines that have a similar angle and position.

    Returns a list of clusters (each cluster is a list of lines).
    """
    clusters = []
    for line in lines:
        angle = line_angle(line)
        midpoint = ((line[0] + line[2]) / 2, (line[1] + line[3]) / 2)
        for cluster in clusters:
            ref = cluster[0]
            diff = abs(angle - line_angle(ref))
            diff = min(diff, 180 - diff)
            if diff < angle_threshold and point_line_distance(midpoint, ref) < distance_threshold:
                cluster.append(line)
                break
        else:
            clusters.append([line])
    return clusters


def calc_center_line(lines):
    """Average lines of a cluster into a single line."""
    ref = np.array(lines[0], dtype=np.float64)
    ref_dir = ref[2:] - ref[:2]
    aligned = []
    for line in lines:
        line = np.array(line, dtype=np.float64)
        # flip endpoints so that every line has the same direction
        if np.dot(line[2:] - line[:2], ref_dir) < 0:
            line = np.concatenate([line[2:], line[:2]])
        aligned.append(line)
    return np.mean(aligned, axis=0).tolist()


def line_intersection(line0, line1, eps=1e-6):
    """Intersection point of two (infinite) lines, or None if parallel."""
    x1, y1, x2, y2 = line0
    x3, y3, x4, y4 = line1
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < eps:
        return None
    det0 = x1 * y2 - y1 * x2
    det1 = x3 * y4 - y3 * x4
    x = (det0 * (x3 - x4) - (x1 - x2) * det1) / denom
    y = (det0 * (y3 - y4) - (y1 - y2) * det1) / denom
    return x, y
