"""Classical (non-learning) Court Detector.

Court-model fitting in the spirit of Farin et al., "Robust Camera Calibration
for Sport Videos using Court Models", 2004.

1. Detect white line pixels with a local contrast test.
2. Detect straight lines (Hough) and split them into horizontal / vertical families.
3. Hypothesize homographies from 2 horizontal + 2 vertical line pairs and score
   each by how well the projected court model overlaps the line pixels.
4. Refine the best homographies by fitting lines to nearby line pixels.
"""

import cv2
import numpy as np

# Court model in the same coordinates as WORLD_COURT_POINTS (~1 cm / pixel).
MODEL_HORIZONTAL = np.array([561, 1110, 2386, 2935], dtype=np.float64)  # baselines, service lines
MODEL_VERTICAL = np.array([286, 423, 832, 1242, 1379], dtype=np.float64)  # sidelines, center line
X_LEFT, X_RIGHT, X_INNER_L, X_INNER_R, X_CENTER = 286, 1379, 423, 1242, 832
Y_TOP, Y_BOTTOM, Y_SERVICE_T, Y_SERVICE_B = 561, 2935, 1110, 2386

# Painted segments of the court: ((x1, y1), (x2, y2)).
MODEL_SEGMENTS = [
    ((X_LEFT, Y_TOP), (X_RIGHT, Y_TOP)),
    ((X_LEFT, Y_BOTTOM), (X_RIGHT, Y_BOTTOM)),
    ((X_INNER_L, Y_SERVICE_T), (X_INNER_R, Y_SERVICE_T)),
    ((X_INNER_L, Y_SERVICE_B), (X_INNER_R, Y_SERVICE_B)),
    ((X_LEFT, Y_TOP), (X_LEFT, Y_BOTTOM)),
    ((X_RIGHT, Y_TOP), (X_RIGHT, Y_BOTTOM)),
    ((X_INNER_L, Y_TOP), (X_INNER_L, Y_BOTTOM)),
    ((X_INNER_R, Y_TOP), (X_INNER_R, Y_BOTTOM)),
    ((X_CENTER, Y_SERVICE_T), (X_CENTER, Y_SERVICE_B)),
]
# Extensions of the painted segments where no line should exist. Image lines that run
# through these (net tape, ad boards, ...) are not court lines. A gap is left near the
# segment ends for the perpendicular lines.
GAP, EXT = 40, 150
X_ALLEY_L, X_ALLEY_R = (X_LEFT + GAP, X_INNER_L - GAP), (X_INNER_R + GAP, X_RIGHT - GAP)
EMPTY_SEGMENTS = [
    # baselines beyond the doubles sidelines
    ((X_LEFT - GAP, Y_TOP), (X_LEFT - EXT, Y_TOP)),
    ((X_RIGHT + GAP, Y_TOP), (X_RIGHT + EXT, Y_TOP)),
    ((X_LEFT - GAP, Y_BOTTOM), (X_LEFT - EXT, Y_BOTTOM)),
    ((X_RIGHT + GAP, Y_BOTTOM), (X_RIGHT + EXT, Y_BOTTOM)),
    # service lines through the alleys
    ((X_ALLEY_L[0], Y_SERVICE_T), (X_ALLEY_L[1], Y_SERVICE_T)),
    ((X_ALLEY_R[0], Y_SERVICE_T), (X_ALLEY_R[1], Y_SERVICE_T)),
    ((X_ALLEY_L[0], Y_SERVICE_B), (X_ALLEY_L[1], Y_SERVICE_B)),
    ((X_ALLEY_R[0], Y_SERVICE_B), (X_ALLEY_R[1], Y_SERVICE_B)),
    # center service line beyond the service lines
    ((X_CENTER, Y_SERVICE_T - GAP), (X_CENTER, Y_SERVICE_T - EXT * 2)),
    ((X_CENTER, Y_SERVICE_B + GAP), (X_CENTER, Y_SERVICE_B + EXT * 2)),
    # sidelines beyond the baselines
    *[((x, Y_TOP - GAP), (x, Y_TOP - EXT)) for x in (X_LEFT, X_RIGHT, X_INNER_L, X_INNER_R)],
    *[((x, Y_BOTTOM + GAP), (x, Y_BOTTOM + EXT)) for x in (X_LEFT, X_RIGHT, X_INNER_L, X_INNER_R)],
]

# Each segment as an infinite line: ('h', y) or ('v', x).
SEGMENT_LINES = [('h', Y_TOP), ('h', Y_BOTTOM), ('h', Y_SERVICE_T), ('h', Y_SERVICE_B),
                 ('v', X_LEFT), ('v', X_RIGHT), ('v', X_INNER_L), ('v', X_INNER_R), ('v', X_CENTER)]

# Same order as WORLD_COURT_POINTS in postprocessor.py.
MODEL_KEYPOINTS = np.array([
    (286, 561), (1379, 561), (286, 2935), (1379, 2935),
    (423, 561), (423, 2935), (1242, 561), (1242, 2935),
    (423, 1110), (1242, 1110), (423, 2386), (1242, 2386),
    (832, 1110), (832, 2386),
], dtype=np.float64)


def sample_segments(segments, step):
    """Sample points along model segments every `step` model pixels."""
    points, owners = [], []
    for i, ((x1, y1), (x2, y2)) in enumerate(segments):
        n = max(2, int(np.hypot(x2 - x1, y2 - y1) / step) + 1)
        t = np.linspace(0, 1, n)
        points.append(np.stack([x1 + (x2 - x1) * t, y1 + (y2 - y1) * t], axis=1))
        owners.append(np.full(n, i))
    return np.concatenate(points), np.concatenate(owners)


def project(H, points):
    """Project Nx2 points with a 3x3 (or Bx3x3) homography."""
    ph = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    out = ph @ np.swapaxes(H, -1, -2)
    w = out[..., 2:3]
    return out[..., :2] / np.where(np.abs(w) < 1e-9, 1e-9, w), w[..., 0]


def homographies_from_points(src, dst):
    """Batched DLT with h33 = 1.

    src, dst: B x 4 x 2
    Returns B x 3 x 3 and a validity mask.
    """
    B = src.shape[0]
    A = np.zeros((B, 8, 8))
    b = np.zeros((B, 8))
    x, y = src[..., 0], src[..., 1]
    u, v = dst[..., 0], dst[..., 1]
    A[:, 0::2, 0] = x
    A[:, 0::2, 1] = y
    A[:, 0::2, 2] = 1
    A[:, 0::2, 6] = -x * u
    A[:, 0::2, 7] = -y * u
    A[:, 1::2, 3] = x
    A[:, 1::2, 4] = y
    A[:, 1::2, 5] = 1
    A[:, 1::2, 6] = -x * v
    A[:, 1::2, 7] = -y * v
    b[:, 0::2] = u
    b[:, 1::2] = v
    valid = np.abs(np.linalg.det(A)) > 1e-12
    A[~valid] = np.eye(8)
    h = np.linalg.solve(A, b[..., None])[..., 0]
    H = np.concatenate([h, np.ones((B, 1))], axis=1).reshape(B, 3, 3)
    return H, valid


def camera_from_homographies(H, img_size):
    """Recover the focal length and camera center from court homographies (batched).

    Assumes square pixels, no skew and the principal point at the image center,
    so that H ~ K [r1 r2 t] with K = diag(f, f, 1).
    Returns f (B, nan when no real solution) and camera centers (B x 3, model units).
    The model z axis points into the ground, so a camera above the court has z < 0.
    """
    H = np.asarray(H, dtype=np.float64).reshape(-1, 3, 3)
    h, w = img_size
    T = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1.0]])
    Hc = T @ H
    a1, b1, c1 = Hc[:, 0, 0], Hc[:, 1, 0], Hc[:, 2, 0]
    a2, b2, c2 = Hc[:, 0, 1], Hc[:, 1, 1], Hc[:, 2, 1]

    # r1 . r2 = 0 and |r1| = |r2| are linear in 1 / f^2; solve them in least squares
    A = np.stack([a1 * a2 + b1 * b2, a1 ** 2 + b1 ** 2 - a2 ** 2 - b2 ** 2], axis=1)
    B = np.stack([c1 * c2, c1 ** 2 - c2 ** 2], axis=1)
    inv_f2 = -(A * B).sum(1) / np.maximum((A * A).sum(1), 1e-30)
    f = np.where(inv_f2 > 0, 1 / np.sqrt(np.abs(inv_f2) + 1e-30), np.nan)

    M = Hc.copy()
    M[:, :2, :] /= f[:, None, None]
    scale = 2 / (np.linalg.norm(M[:, :, 0], axis=1) + np.linalg.norm(M[:, :, 1], axis=1))
    M *= (scale * np.sign(M[:, 2, 2]))[:, None, None]  # the court is in front of the camera
    r1, r2, t = M[:, :, 0], M[:, :, 1], M[:, :, 2]
    R = np.stack([r1, r2, np.cross(r1, r2)], axis=2)
    center = -np.einsum('bji,bj->bi', R, t)
    return f, center


def intersect(l0, l1):
    """Intersection of lines in homogeneous form (a, b, c): ax + by + c = 0."""
    p = np.cross(l0, l1)
    return p[..., :2] / p[..., 2:3]


class ClassicalCourtDetector:
    """Detect 14 court keypoints without learning."""

    def __init__(self,
                 line_tau=(8, 4),
                 line_min_brightness=40,
                 line_min_contrast=8,
                 line_rel_contrast=0.15,
                 coherence_sigma=3.0,
                 min_coherence=0.5,
                 hough_threshold=50,
                 min_line_length=60,
                 max_line_gap=15,
                 max_lines=15,
                 score_tolerance=3,
                 refine_band=8,
                 refine_iters=3,
                 top_k=10,
                 empty_penalty=1.0,
                 focal_range=(0.5, 6.0),
                 camera_height_range=(150, 3000)):
        """Init.

        - line_tau: (horizontal, vertical) offset of the local contrast test.
          Should be larger than the line width in pixels.
        - line_min_brightness / line_min_contrast: thresholds of the contrast test.
        - line_rel_contrast: contrast threshold relative to the neighbor brightness.
        - coherence_sigma / min_coherence: structure tensor filter that keeps linear structures.
        - max_lines: number of candidate lines kept per family.
        - score_tolerance: pixel tolerance when scoring a homography.
        - refine_band: half width of the search band when refining lines.
        - focal_range: allowed focal length in units of the image width. None disables the camera filter.
        - camera_height_range: allowed camera height in model units (~cm).
        - empty_penalty: penalty for line pixels on EMPTY_SEGMENTS (e.g. net tape matched as a baseline).
        """
        self.line_tau = line_tau
        self.line_min_brightness = line_min_brightness
        self.line_min_contrast = line_min_contrast
        self.line_rel_contrast = line_rel_contrast
        self.coherence_sigma = coherence_sigma
        self.min_coherence = min_coherence
        self.hough_threshold = hough_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.max_lines = max_lines
        self.score_tolerance = score_tolerance
        self.refine_band = refine_band
        self.refine_iters = refine_iters
        self.top_k = top_k
        self.empty_penalty = empty_penalty
        self.focal_range = focal_range
        self.camera_height_range = camera_height_range

        self.coarse_points, self.coarse_owners = sample_segments(MODEL_SEGMENTS, step=40)
        self.fine_points, self.fine_owners = sample_segments(MODEL_SEGMENTS, step=10)
        self.coarse_empty_points, _ = sample_segments(EMPTY_SEGMENTS, step=20)
        self.fine_empty_points, _ = sample_segments(EMPTY_SEGMENTS, step=10)
        self.matrix = None
        self.score = 0.0

    def detect_line_pixels(self, img):
        """Binary mask of white line pixels.

        A pixel belongs to a line when it is bright and the pixels at distance tau
        on both sides (horizontally or vertically) are darker. The contrast threshold
        is relative to the neighbors so that lines in shadow are kept.

        img: RGB numpy image (h, w, 3)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.int16)
        tx, ty = self.line_tau
        c = self.line_min_contrast
        r = self.line_rel_contrast

        def darker_both_sides(axis, tau):
            fwd = np.roll(gray, tau, axis=axis)
            bwd = np.roll(gray, -tau, axis=axis)
            return (gray - fwd > np.maximum(c, r * fwd)) & (gray - bwd > np.maximum(c, r * bwd))

        mask = (gray > self.line_min_brightness) & (darker_both_sides(1, tx) | darker_both_sides(0, ty))

        # keep only locally linear structures (rejects crowd, text and textures)
        g = gray.astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        sigma = self.coherence_sigma
        jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
        jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
        jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
        coherence = ((jxx - jyy) ** 2 + 4 * jxy ** 2) / ((jxx + jyy) ** 2 + 1e-6)
        mask &= coherence > self.min_coherence
        return mask.astype(np.uint8) * 255

    def detect_lines(self, mask):
        """Detect candidate lines and split them into horizontal / vertical families.

        Returns two arrays of homogeneous lines (N x 3), normalized so that a^2 + b^2 = 1.
        """
        segs = cv2.HoughLinesP(mask, 1, np.pi / 180, self.hough_threshold,
                               minLineLength=self.min_line_length, maxLineGap=self.max_line_gap)
        if segs is None:
            return np.zeros((0, 3)), np.zeros((0, 3))
        segs = segs.reshape(-1, 4).astype(np.float64)

        # merge collinear segments
        clusters = []
        for seg in sorted(segs, key=lambda s: -np.hypot(s[2] - s[0], s[3] - s[1])):
            line = self.segment_to_line(seg)
            length = np.hypot(seg[2] - seg[0], seg[3] - seg[1])
            for cluster in clusters:
                ref = cluster['line']
                mid0 = seg[:2]
                mid1 = seg[2:]
                if abs(ref[:2] @ line[:2]) > np.cos(np.radians(2)) \
                        and abs(ref @ [*mid0, 1]) < 6 and abs(ref @ [*mid1, 1]) < 6:
                    cluster['segs'].append(seg)
                    cluster['length'] += length
                    break
            else:
                clusters.append({'line': line, 'segs': [seg], 'length': length})

        lines = []
        for cluster in clusters:
            pts = np.concatenate([s.reshape(2, 2) for s in cluster['segs']])
            line = self.fit_line(pts) if len(cluster['segs']) > 1 else cluster['line']
            lines.append((line, cluster['length']))

        horizontal, vertical = [], []
        for line, length in lines:
            # normal (a, b): a horizontal line has |b| ~ 1
            angle = np.degrees(np.arctan2(abs(line[0]), abs(line[1])))  # 0 = horizontal
            if angle < 20:
                horizontal.append((line, length))
            elif angle > 25:
                vertical.append((line, length))

        def top(lst):
            lst = sorted(lst, key=lambda t: -t[1])[:self.max_lines]
            return np.array([t[0] for t in lst]).reshape(-1, 3)

        return top(horizontal), top(vertical)

    @staticmethod
    def segment_to_line(seg):
        """Homogeneous normalized line through a segment."""
        x1, y1, x2, y2 = seg
        line = np.cross([x1, y1, 1.0], [x2, y2, 1.0])
        return line / np.hypot(line[0], line[1])

    @staticmethod
    def fit_line(points, weights=None):
        """Total least squares line fit. Returns a normalized homogeneous line."""
        points = np.asarray(points, dtype=np.float64)
        if weights is None:
            weights = np.ones(len(points))
        mean = (points * weights[:, None]).sum(0) / weights.sum()
        centered = (points - mean) * np.sqrt(weights[:, None])
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        normal = vt[-1]
        return np.array([normal[0], normal[1], -normal @ mean])

    def hypothesize(self, horizontal, vertical, img_size):
        """All homographies from 2 horizontal + 2 vertical line correspondences."""
        h, w = img_size
        if len(horizontal) < 2 or len(vertical) < 2:
            return np.zeros((0, 3, 3))

        # order image lines: horizontal by y at the image center, vertical by x at the image center
        cx, cy = w / 2, h / 2
        hy = -(horizontal[:, 0] * cx + horizontal[:, 2]) / horizontal[:, 1]
        vx = -(vertical[:, 1] * cy + vertical[:, 2]) / vertical[:, 0]
        horizontal = horizontal[np.argsort(hy)]
        vertical = vertical[np.argsort(vx)]

        hi, hj = np.triu_indices(len(horizontal), 1)
        vi, vj = np.triu_indices(len(vertical), 1)
        mhi, mhj = np.triu_indices(len(MODEL_HORIZONTAL), 1)
        mvi, mvj = np.triu_indices(len(MODEL_VERTICAL), 1)
        # model vertical lines must be ordered left-to-right
        order = np.argsort(MODEL_VERTICAL)
        mv_sorted = MODEL_VERTICAL[order]

        # image intersections for every (h pair, v pair)
        H0, H1 = horizontal[hi], horizontal[hj]
        V0, V1 = vertical[vi], vertical[vj]
        img_pts = np.stack([
            intersect(H0[:, None], V0[None]),
            intersect(H0[:, None], V1[None]),
            intersect(H1[:, None], V0[None]),
            intersect(H1[:, None], V1[None]),
        ], axis=2)  # Ph x Pv x 4 x 2

        my0, my1 = MODEL_HORIZONTAL[mhi], MODEL_HORIZONTAL[mhj]
        mx0, mx1 = mv_sorted[mvi], mv_sorted[mvj]
        model_pts = np.stack([
            np.stack(np.broadcast_arrays(mx0[None], my0[:, None]), -1),
            np.stack(np.broadcast_arrays(mx1[None], my0[:, None]), -1),
            np.stack(np.broadcast_arrays(mx0[None], my1[:, None]), -1),
            np.stack(np.broadcast_arrays(mx1[None], my1[:, None]), -1),
        ], axis=2)  # Mh x Mv x 4 x 2

        Ph, Pv = img_pts.shape[:2]
        Mh, Mv = model_pts.shape[:2]
        src = np.broadcast_to(model_pts[None, :, None, :], (Ph, Mh, Pv, Mv, 4, 2)).reshape(-1, 4, 2)
        dst = np.broadcast_to(img_pts[:, None, :, None], (Ph, Mh, Pv, Mv, 4, 2)).reshape(-1, 4, 2)
        finite = np.isfinite(dst).all(axis=(1, 2)) & (np.abs(dst) < 1e5).all(axis=(1, 2))
        H, valid = homographies_from_points(src[finite], dst[finite])
        return H[valid]

    def plausible(self, H, img_size):
        """Reject degenerate homographies."""
        h, w = img_size
        corners = MODEL_KEYPOINTS[:4]  # TL, TR, BL, BR
        proj, depth = project(H, corners)
        ok = (depth > 0).all(axis=1)
        tl, tr, bl, br = proj[:, 0], proj[:, 1], proj[:, 2], proj[:, 3]
        # broadcast view: far baseline above near baseline, left of right, near baseline wider
        ok &= (tl[:, 0] < tr[:, 0]) & (bl[:, 0] < br[:, 0])
        ok &= (tl[:, 1] < bl[:, 1]) & (tr[:, 1] < br[:, 1])
        far_w = tr[:, 0] - tl[:, 0]
        near_w = br[:, 0] - bl[:, 0]
        ok &= (near_w >= far_w * 0.95) & (near_w < far_w * 4)
        ok &= (far_w > w * 0.1) & (near_w < w * 4)
        ok &= (bl[:, 1] - tl[:, 1] > h * 0.15)

        # physically plausible broadcast camera (about twice the range seen in the dataset)
        if self.focal_range is None:
            return ok
        f, center = camera_from_homographies(H, img_size)
        f_min, f_max = self.focal_range
        z_min, z_max = self.camera_height_range
        ok &= (f > f_min * w) & (f < f_max * w)
        ok &= (-center[:, 2] > z_min) & (-center[:, 2] < z_max)
        ok &= center[:, 1] > Y_BOTTOM
        return ok

    def score_homographies(self, H, dist_map, points, empty_points=None, chunk=20000):
        """Score = hits on line pixels - penalty for misses - penalty for lines where none should be.

        points: samples on painted model segments
        empty_points: samples on model regions where no line should exist
        """
        h, w = dist_map.shape
        num_points = len(points)
        if empty_points is not None:
            points = np.concatenate([points, empty_points])
        scores = np.empty(len(H))
        for s in range(0, len(H), chunk):
            proj, _ = project(H[s:s + chunk], points)
            x = np.round(proj[..., 0]).astype(np.int64)
            y = np.round(proj[..., 1]).astype(np.int64)
            inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            d = np.full(x.shape, np.inf)
            d[inside] = dist_map[y[inside], x[inside]]
            hit = d <= self.score_tolerance
            miss = inside & ~hit
            scores[s:s + chunk] = hit[:, :num_points].sum(1) - 0.5 * miss[:, :num_points].sum(1) \
                - self.empty_penalty * hit[:, num_points:].sum(1)
        return scores

    def refine(self, H, mask_points, img_size):
        """Refine a homography by re-fitting each model line to nearby line pixels."""
        h, w = img_size
        for _ in range(self.refine_iters):
            fitted = {}
            for i, segment in enumerate(MODEL_SEGMENTS):
                seg = np.array(segment, dtype=np.float64)
                (p0, p1), _ = project(H, seg)
                if not np.all(np.isfinite([p0, p1])):
                    continue
                line = self.segment_to_line([*p0, *p1])
                d = mask_points @ line[:2] + line[2]
                # restrict to the segment extent
                direction = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
                t = (mask_points - p0) @ direction
                near = (np.abs(d) < self.refine_band) & (t > 0) & (t < np.linalg.norm(p1 - p0))
                if near.sum() < 30:
                    continue
                pts = mask_points[near]
                # robust: a second pass around the first fit
                fit = self.fit_line(pts)
                pts = pts[np.abs(pts @ fit[:2] + fit[2]) < 2.5]
                if len(pts) < 20:
                    continue
                fitted[i] = self.fit_line(pts)

            # intersections of fitted horizontal and vertical lines
            src, dst = [], []
            h_ids = [i for i in fitted if SEGMENT_LINES[i][0] == 'h']
            v_ids = [i for i in fitted if SEGMENT_LINES[i][0] == 'v']
            for i in h_ids:
                for j in v_ids:
                    p = intersect(fitted[i], fitted[j])
                    if np.all(np.isfinite(p)) and np.all(np.abs(p) < 1e5):
                        src.append((SEGMENT_LINES[j][1], SEGMENT_LINES[i][1]))
                        dst.append(p)
            if len(src) < 4 or len(h_ids) < 2 or len(v_ids) < 2:
                break
            H_new, _ = cv2.findHomography(np.array(src, dtype=np.float64),
                                          np.array(dst, dtype=np.float64), 0)
            if H_new is None:
                break
            H = H_new
        return H

    def __call__(self, img):
        """Detect court keypoints.

        img: RGB numpy image (h, w, 3)
        Returns keypoints (14 x 2 list) and scores (14 list).
        Score is 1 for a keypoint inside the image and 0 otherwise (or when detection fails).
        """
        img_size = img.shape[:2]
        h, w = img_size
        mask = self.detect_line_pixels(img)
        horizontal, vertical = self.detect_lines(mask)

        H = self.hypothesize(horizontal, vertical, img_size)
        if len(H):
            H = H[self.plausible(H, img_size)]
        if len(H) == 0:
            self.matrix, self.score = None, 0.0
            return np.zeros((14, 2)).tolist(), [0.0] * 14

        dist_map = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
        scores = self.score_homographies(H, dist_map, self.coarse_points, self.coarse_empty_points)
        top = np.argsort(-scores)[:self.top_k]

        ys, xs = np.nonzero(mask)
        mask_points = np.stack([xs, ys], axis=1).astype(np.float64)
        best_H, best_score = None, -np.inf
        for k in top:
            H_ref = self.refine(H[k], mask_points, img_size)
            score = self.score_homographies(H_ref[None], dist_map, self.fine_points, self.fine_empty_points)[0]
            if score > best_score:
                best_H, best_score = H_ref, score

        self.matrix = best_H
        self.score = best_score / len(self.fine_points)
        keypoints, _ = project(best_H, MODEL_KEYPOINTS)
        inside = (keypoints[:, 0] >= 0) & (keypoints[:, 0] < w) \
            & (keypoints[:, 1] >= 0) & (keypoints[:, 1] < h)
        return keypoints.tolist(), inside.astype(np.float64).tolist()
