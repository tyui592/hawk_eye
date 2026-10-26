"""Classical (non-learning) Ball Tracker.

Candidate generation + trajectory selection in the spirit of Yan, Christmas & Kittler,
"A Tennis Ball Tracking Algorithm for Automatic Annotation of Tennis Match", 2005.

1. Candidates: 3-frame signed differencing (the ball is brighter than the background it
   covers in both neighbouring frames), optionally after aligning the neighbours to the
   current frame (ORB + RANSAC) to cancel small camera motion.
2. Seeds: triplets of candidates in frames (t-1, t, t+1) that move with nearly constant
   velocity.
3. Tracklets: grow each seed forward and backward with a constant-velocity prediction,
   allowing short gaps.
4. Selection: long tracklets claim frames first; short gaps inside a tracklet are
   linearly interpolated.
"""

import cv2
import numpy as np


class ClassicalBallTracker:
    """Track a tennis ball in a clip from a fixed (broadcast) camera."""

    def __init__(self, diff_threshold=15, min_area=3, max_area=400, max_raw=300, isolation_radius=40,
                 region_threshold=10, region_dilate=9, region_area=800, top_k=10, align=False,
                 seed_gate=80, seed_accel=8, grow_radius=12, grow_radius_ratio=0.5,
                 max_gap=4, min_track_len=6, min_track_speed=2.5, speed_ref=15,
                 chain=False, chain_gap=15, chain_overlap=2, chain_radius=30, chain_speed=40,
                 velocity_offset=0.1):
        """Initialize with parameters.

        - diff_threshold: minimum gray-level difference against both neighbours.
        - isolation_radius: candidates closer than this count as neighbours.
        - region_*: quality is scaled by min(1, region_area / area of the dilated |diff|
          component around the candidate).
        - top_k: candidates kept per frame (by quality).
        - seed_gate: maximum displacement (px) between consecutive seed points.
        - seed_accel: maximum |p0 - 2 p1 + p2| (px) of a seed triplet.
        - grow_radius(+ratio * speed): search radius around the predicted position.
        - max_gap: frames a tracklet may skip without a candidate.
        - min_track_len: minimum number of observed points of a selected tracklet.
        - min_track_speed: tracklets slower than this median speed (px/frame) are dropped.
        - speed_ref: tracklet scores are scaled by min(1, median speed / speed_ref).
        - chain*: keep only the best spatio-temporally consistent chain of tracklets (`_chain`).
          Off by default: it lowered recall on game9 (0.80 -> 0.69).
        - velocity_offset: output position is shifted by this fraction of the velocity.
        """
        self.diff_threshold = diff_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.max_raw = max_raw
        self.isolation_radius = isolation_radius
        self.region_threshold = region_threshold
        self.region_dilate = region_dilate
        self.region_area = region_area
        self.top_k = top_k
        self.align = align
        self.seed_gate = seed_gate
        self.seed_accel = seed_accel
        self.grow_radius = grow_radius
        self.grow_radius_ratio = grow_radius_ratio
        self.max_gap = max_gap
        self.min_track_len = min_track_len
        self.min_track_speed = min_track_speed
        self.speed_ref = speed_ref
        self.chain = chain
        self.chain_gap = chain_gap
        self.chain_overlap = chain_overlap
        self.chain_radius = chain_radius
        self.chain_speed = chain_speed
        self.velocity_offset = velocity_offset
        if align:
            self.orb = cv2.ORB_create(1000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # ------------------------------------------------------------------ candidates
    def _warp_to(self, src, dst):
        """Warp `src` onto `dst` with a homography estimated from ORB matches."""
        kp1, des1 = self.orb.detectAndCompute(src, None)
        kp2, des2 = self.orb.detectAndCompute(dst, None)
        if des1 is None or des2 is None:
            return src
        matches = self.matcher.match(des1, des2)
        if len(matches) < 20:
            return src
        p1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        p2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        H, _ = cv2.findHomography(p1, p2, cv2.RANSAC, 2.0)
        if H is None:
            return src
        return cv2.warpPerspective(src, H, (dst.shape[1], dst.shape[0]), borderMode=cv2.BORDER_REPLICATE)

    def detect_candidates(self, prev, cur, nxt):
        """Ball candidates in `cur` from gray frames.

        Output: (k, 5) array of [x, y, peak response, area, quality], sorted by quality.
        """
        if self.align:
            prev, nxt = self._warp_to(prev, cur), self._warp_to(nxt, cur)
        c = cur.astype(np.int16)
        resp = np.minimum(c - prev, c - nxt).clip(0, 255).astype(np.uint8)
        mask = (resp > self.diff_threshold).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return np.zeros((0, 5))

        # response-weighted centroid and peak response per component
        ys, xs = np.nonzero(labels)
        lab = labels[ys, xs]
        w = resp[ys, xs].astype(np.float64)
        wsum = np.bincount(lab, w, n)
        cx = np.bincount(lab, w * xs, n) / np.maximum(wsum, 1e-9)
        cy = np.bincount(lab, w * ys, n) / np.maximum(wsum, 1e-9)
        peak = np.zeros(n)
        np.maximum.at(peak, lab, w)
        area = stats[:, cv2.CC_STAT_AREA]

        keep = np.arange(1, n)
        keep = keep[(area[keep] >= self.min_area) & (area[keep] <= self.max_area)]
        keep = keep[np.argsort(-peak[keep])][:self.max_raw]
        if len(keep) == 0:
            return np.zeros((0, 5))

        # The ball is isolated; moving players break into clusters of small blobs.
        x, y = cx[keep], cy[keep]
        neighbours = (np.hypot(x[:, None] - x[None], y[:, None] - y[None]) < self.isolation_radius).sum(1) - 1
        quality = peak[keep] / (1 + neighbours)

        # A small far player can still split into isolated blobs (head, shoes), but the
        # whole moving region around it is much larger than the ball's.
        motion = (cv2.absdiff(cur, prev) > self.region_threshold) | (cv2.absdiff(cur, nxt) > self.region_threshold)
        motion = cv2.dilate(motion.astype(np.uint8), np.ones((self.region_dilate, self.region_dilate), np.uint8))
        _, rlabels, rstats, _ = cv2.connectedComponentsWithStats(motion)
        h, w = cur.shape
        region = rstats[rlabels[np.clip(y.round().astype(int), 0, h - 1), np.clip(x.round().astype(int), 0, w - 1)],
                        cv2.CC_STAT_AREA]
        quality = quality * np.minimum(1.0, self.region_area / np.maximum(region, 1))
        cands = np.stack([x, y, peak[keep], area[keep], quality], axis=1)
        return cands[np.argsort(-quality)][:self.top_k]

    # ------------------------------------------------------------------ tracking
    def _seeds(self, cands):
        """Constant-velocity triplets: list of (cost, t, i, j, k)."""
        seeds = []
        for t in range(1, len(cands) - 1):
            a, b, c = cands[t - 1][:, :2], cands[t][:, :2], cands[t + 1][:, :2]
            if len(a) == 0 or len(b) == 0 or len(c) == 0:
                continue
            ab = np.linalg.norm(b[None, :, :] - a[:, None, :], axis=2)  # (A, B)
            bc = np.linalg.norm(c[None, :, :] - b[:, None, :], axis=2)  # (B, C)
            acc = np.linalg.norm(a[:, None, None, :] - 2 * b[None, :, None, :] + c[None, None, :, :], axis=3)
            ok = (ab[:, :, None] < self.seed_gate) & (bc[None, :, :] < self.seed_gate) \
                & (acc < self.seed_accel) & (ab[:, :, None] > 1) & (bc[None, :, :] > 1)
            for i, j, k in zip(*np.nonzero(ok)):
                seeds.append((acc[i, j, k], t, i, j, k))
        seeds.sort()
        return seeds

    def _grow(self, cands, used, track, direction):
        """Extend `track` ({t: (x, y, quality)}) in `direction` (+1 / -1) with a constant-velocity model."""
        ts = sorted(track)
        if direction < 0:
            ts = ts[::-1]
        t1, t0 = ts[-1], ts[-2]
        p1, p0 = np.array(track[t1][:2]), np.array(track[t0][:2])
        vel = (p1 - p0) / (t1 - t0)
        t, gap = t1, 0
        while gap <= self.max_gap:
            t += direction
            if t < 0 or t >= len(cands):
                break
            steps = abs(t - t1)
            pred = p1 + vel * steps * direction
            radius = self.grow_radius + self.grow_radius_ratio * np.linalg.norm(vel) * steps
            cs = cands[t]
            best = None
            if len(cs):
                d = np.linalg.norm(cs[:, :2] - pred, axis=1)
                d[[i for i in range(len(cs)) if (t, i) in used]] = np.inf
                i = int(np.argmin(d))
                if d[i] < radius:
                    best = i
            if best is None:
                gap += 1
                continue
            p = cs[best, :2].copy()
            vel = (p - p1) / steps * direction
            track[t] = (p[0], p[1], cs[best, 4])
            used.add((t, best))
            p1, t1, gap = p, t, 0
        return track

    def _chain(self, scored):
        """Keep the maximum-score chain of tracklets that the ball could have followed.

        A tracklet B may follow A when B starts after A ends (up to `chain_overlap` frames
        earlier). When the gap is at most `chain_gap` frames, B's first point must also be
        reachable from A's last point (`chain_radius + chain_speed * gap` px); longer gaps
        are free, so the chain can restart after the ball disappears.
        """
        items = []
        for score, track in scored:
            ts = sorted(track)
            items.append((ts[0], ts[-1], np.array(track[ts[0]][:2]), np.array(track[ts[-1]][:2]), score, track))
        items.sort(key=lambda it: it[0])
        n = len(items)
        best = np.zeros(n)
        prev = [-1] * n
        for b in range(n):
            sb, _, pb, _, wb, _ = items[b]
            best[b], prev[b] = wb, -1
            for a in range(b):
                _, ea, _, qa, _, _ = items[a]
                gap = sb - ea
                if gap < -self.chain_overlap:
                    continue
                if gap <= self.chain_gap and \
                        np.linalg.norm(pb - qa) > self.chain_radius + self.chain_speed * max(gap, 1):
                    continue
                if best[a] + wb > best[b]:
                    best[b], prev[b] = best[a] + wb, a
        if n == 0:
            return []
        chain, b = [], int(np.argmax(best))
        while b >= 0:
            chain.append((items[b][4], items[b][5]))
            b = prev[b]
        return chain

    def track(self, cands):
        """Select ball positions from per-frame candidates.

        Output: list of (x, y) or None per frame, list of scores.
        """
        num = len(cands)
        used = set()
        tracks = []
        for _, t, i, j, k in self._seeds(cands):
            if (t - 1, i) in used or (t, j) in used or (t + 1, k) in used:
                continue
            track = {s: tuple(cands[s][c, [0, 1, 4]]) for s, c in ((t - 1, i), (t, j), (t + 1, k))}
            claimed = {(t - 1, i), (t, j), (t + 1, k)}
            track = self._grow(cands, claimed, track, +1)
            track = self._grow(cands, claimed, track, -1)
            if len(track) >= self.min_track_len:
                used |= claimed
                tracks.append(track)

        # Tracklets with the largest score claim frames first. Players move slower than the
        # ball, so slow tracklets are down-weighted or dropped.
        scored = []
        for track in tracks:
            ts = sorted(track)
            p = np.array([track[t][:2] for t in ts])
            speed = np.median(np.linalg.norm(np.diff(p, axis=0), axis=1) / np.diff(ts))
            if speed < self.min_track_speed:
                continue
            score = sum(v[2] for v in track.values()) * min(1.0, speed / self.speed_ref)
            scored.append((score, track))
        if self.chain:
            scored = self._chain(scored)
        scored.sort(key=lambda s: -s[0])

        positions, scores = [None] * num, [0.0] * num
        for _, track in scored:
            ts = sorted(track)
            free = [t for t in range(ts[0], ts[-1] + 1) if positions[t] is None]
            if sum(t in track for t in free) < self.min_track_len:
                continue
            for t in free:
                if t in track:
                    positions[t], scores[t] = track[t][:2], 1.0
                else:
                    t0 = max(s for s in ts if s < t)
                    t1 = min(s for s in ts if s > t)
                    r = (t - t0) / (t1 - t0)
                    p0, p1 = np.array(track[t0][:2]), np.array(track[t1][:2])
                    positions[t], scores[t] = tuple(p0 + r * (p1 - p0)), 0.75

        # The differencing response lags behind the labelled centre along the motion.
        if self.velocity_offset:
            shifted = list(positions)
            for t in range(1, num - 1):
                if positions[t - 1] is not None and positions[t] is not None and positions[t + 1] is not None:
                    v = (np.array(positions[t + 1]) - np.array(positions[t - 1])) / 2
                    shifted[t] = tuple(np.array(positions[t]) + self.velocity_offset * v)
            positions = shifted
        return positions, scores

    def __call__(self, frames):
        """Track the ball in a list of BGR or gray frames.

        Output: positions [(x, y) or None], scores [float], candidates [(k, 4) array].
        """
        gray = [f if f.ndim == 2 else cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        cands = [np.zeros((0, 5))]
        for t in range(1, len(gray) - 1):
            cands.append(self.detect_candidates(gray[t - 1], gray[t], gray[t + 1]))
        cands.append(np.zeros((0, 5)))
        positions, scores = self.track(cands)
        return positions, scores, cands
