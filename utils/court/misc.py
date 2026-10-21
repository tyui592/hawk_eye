"""Court Utilities."""

import cv2
import numpy as np

# Court reference image (same coordinates as WORLD_COURT_POINTS, ~1 cm / pixel).
COURT_IMG_SIZE = (1665, 3496)  # width, height
COURT_LINES = [
    ((286, 561), (1379, 561)),     # top baseline
    ((286, 2935), (1379, 2935)),   # bottom baseline
    ((286, 1748), (1379, 1748)),   # net
    ((286, 561), (286, 2935)),     # left doubles sideline
    ((1379, 561), (1379, 2935)),   # right doubles sideline
    ((423, 561), (423, 2935)),     # left singles sideline
    ((1242, 561), (1242, 2935)),   # right singles sideline
    ((423, 1110), (1242, 1110)),   # top service line
    ((423, 2386), (1242, 2386)),   # bottom service line
    ((832, 1110), (832, 2386)),    # center service line
]


def get_court_img(line_width=10):
    """Get a court reference image (BGR, white lines on black)."""
    width, height = COURT_IMG_SIZE
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for pt1, pt2 in COURT_LINES:
        cv2.line(img, pt1, pt2, (255, 255, 255), line_width)
    return img
