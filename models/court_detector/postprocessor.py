"""Postprocessing of Court Detection."""

import cv2
import torch
import numpy as np
import torch.nn.functional as F
from utils.court.line_ops import cluster_lines, calc_center_line, line_intersection

# World Court Points
WORLD_COURT_POINTS = [
    (286, 561),
    (1379, 561),
    (286, 2935),
    (1379, 2935),
    (423, 561),
    (423, 2935),
    (1242, 561),
    (1242, 2935),
    (423, 1110),
    (1242, 1110),
    (423, 2386),
    (1242, 2386),
    (832, 1110),
    (832, 2386),
]

RECTANGLE_INDICES = [
    (0, 4, 2, 5),
    (0, 6, 2, 7),
    (0, 1, 2, 3),
    (4, 6, 5, 7),
    (4, 1, 5, 3),
    (8, 12, 10, 13),
    (8, 9, 10, 11),
    (4, 6, 8, 9),
    (4, 6, 10, 11),
    (8, 9, 5, 7),
    (6, 1, 7, 3),
    (10, 11, 5, 7),
    (12, 9, 13, 11),
    list(range(14)),
]


class CourtPostprocessor:
    """Court edge postprocessor for refine predicted keypontis.

    Refine predicted keypoints with image processing and homography transform.
    """

    def __init__(self,
                 refine_flag=0,
                 linecolor_threshold=155,
                 hough_threshold=20,
                 crop_size=20,
                 homography_threshold=0.9,
                 homography_method=cv2.RANSAC,
                 ransac_threshold=0.4):
        """Initialize with parameters.
        
        - Method 1. Refine with lines from the image.
            - linecolor_threshold: To binarize the image processing.
            - hough_threshold: For hough line detection.
        - MEthod 2. Refine with homography transform.
            - homography_threshold: reprojection error(keypoint distance) threshold
        """
        self.refine_flag = refine_flag
        self.hough_threshold = hough_threshold
        self.linecolor_threshold = linecolor_threshold
        self.crop_size = crop_size
        self.near_net_points = [8, 9, 12]
        self.homography_threshold = homography_threshold
        self.homography_method = homography_method
        self.ransac_threshold = ransac_threshold
        self.matrix = None

    @torch.no_grad()
    def heatmap2points(self, outputs, raw_size, kernel_size=3):
        """Postprocess the model outputs.
        
        Input
        - outputs: bs, layers, channel, height, width

        Output
        - keypoints: bs, channel, 2
        - scores: bs, channel

        """
        heatmap = outputs.unbind(1)[-1].sigmoid()
        maxpool = F.max_pool2d(heatmap, kernel_size, 1, kernel_size // 2)
        is_peak = heatmap == maxpool
        peakmap = heatmap * is_peak
        scores, max_indices = torch.max(peakmap.view(peakmap.size(0), peakmap.size(1), -1), dim=-1)

        height, width = peakmap.size(2), peakmap.size(3)
        y_coords = max_indices // width  # height 좌표
        x_coords = max_indices % width   # width 좌표

        # scaling to image scale
        raw_h, raw_w = torch.tensor(raw_size).unbind(1)
        keypoints = torch.stack([
            x_coords.cpu() * raw_w.unsqueeze(1) / width,
            y_coords.cpu() * raw_h.unsqueeze(1) / height,
        ], dim=-1)
        
        return keypoints.tolist(), scores.cpu().tolist()

    def detect_lines(self, img):
        """Detect lines in image.

        img: a numpy image(h, w, c)
        """
        img_h, img_w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.threshold(gray, self.linecolor_threshold, 255, cv2.THRESH_BINARY)[1]
        min_line_length = min(img_h, img_w) // 4
        lines = cv2.HoughLinesP(edges,
                                1,
                                np.pi / 180,
                                threshold=self.hough_threshold,
                                minLineLength=min_line_length,
                                maxLineGap=min_line_length * 2)
        if lines is None:
            return []
        return lines[:, 0, :].tolist()
    
    def __call__(self, outputs, inputs):
        keypoints, scores = self.heatmap2points(outputs, inputs['raw_size'])

        if self.refine_flag > 0:
            if self.refine_flag == 1:
                refine_with_line = True
                refine_with_homography = False
            elif self.refine_flag == 2:
                refine_with_line = True
                refine_with_homography = True
            elif self.refine_flag == 3:
                refine_with_line = False
                refine_with_homography = True
            keypoints = self.refine_keypoints(inputs['raw_img'],
                                              keypoints,
                                              refine_with_line,
                                              refine_with_homography)
        
        return keypoints, scores

    def refine_keypoints(self,
                 images: list[np.ndarray],
                 keypoints: list[list],
                 refine_with_line: bool = True,
                 refine_with_homography: bool = True):
        """Postprocess with network's output keypoints.
        image: C, H, W
        keypoitns: N x 2 (model prediction)
        """
        res = []
        for img, kps in zip(images, keypoints):
            if refine_with_line:
                kps = self.refine_with_lines(img, kps)

            if refine_with_homography:
                kps, _ = self.refine_with_homography(kps)
            res.append(kps)
        return res
    
    def refine_with_lines(self, img, points):
        """Refine with line detections.

        points: N x 2
        labels: N
        """
        img_h, img_w = img.shape[:2]
        res = []
        for point_index, point in enumerate(points):
            x, y = map(round, point)

            crop_xmin = max(0, x - self.crop_size)
            crop_ymin = max(0, y - self.crop_size)
            crop_xmax = min(img_w, x + self.crop_size)
            crop_ymax = min(img_h, y + self.crop_size)
            if point_index in self.near_net_points:
                crop_ymax = min(img_h, y + self.crop_size // 4)
            img_crop = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax].copy()

            lines = self.detect_lines(img_crop)
            if lines:
                clusters = cluster_lines(lines)
                center_lines = []
                for cluster in clusters:
                    if len(cluster) > 1:
                        center_lines.append(calc_center_line(cluster))
                    else:
                        center_lines.append(cluster[0])

                if len(center_lines) == 2:
                    line0, line1 = center_lines
                    intersection = line_intersection(line0, line1)
                    if intersection is not None:
                        dx, dy = intersection
                        if (0 <= dx < crop_xmax - crop_xmin) \
                            and (0 <= dy < crop_ymax - crop_ymin):
                            x = crop_xmin + dx
                            y = crop_ymin + dy
            res.append((x, y))
        return res

    def refine_with_homography(self, points):
        """Get edge points with homography matrix."""
        reference_world_points = np.array(WORLD_COURT_POINTS, dtype=np.float32)[:, None, :]
        court_image_points = np.array(points, dtype=np.float32)[:, None, :]
        min_dist = float('inf')
        res = points
        for indices in RECTANGLE_INDICES:
            points0 = np.stack([reference_world_points[i] for i in indices], axis=0)
            points1 = np.stack([court_image_points[i] for i in indices], axis=0)

            # calculate projection matrix
            matrix, _ = cv2.findHomography(points0,
                                           points1,
                                           method=self.homography_method,
                                           ransacReprojThreshold=self.ransac_threshold)

            # project world to image
            reference_image_points = cv2.perspectiveTransform(reference_world_points, matrix)

            # measure pixel distance
            dist = np.mean(np.abs(reference_image_points - court_image_points))
            
            if dist < min_dist: 
                min_dist = dist
                self.matrix = matrix
                if  min_dist < self.homography_threshold:
                    res = reference_image_points[:, 0, :].tolist()

        return res, min_dist


def get_court_postprocessor(args):
    postprocessor = CourtPostprocessor(refine_flag=args.refine_flag)
    return postprocessor
                                       