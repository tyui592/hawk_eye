"""Postprocessing of Ball Detection."""

import torch
import torch.nn.functional as F

class BallPostprocessor:
    """Court edge postprocessor for refine predicted keypontis.

    Refine predicted keypoints with image processing and homography transform.
    """

    def __init__(self):
        """Initialize with parameters."""

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

        height, width = peakmap.shape[2:]
        y_coords = max_indices // width  # height 좌표
        x_coords = max_indices % width   # width 좌표

        # scaling to image scale
        raw_h, raw_w = torch.tensor(raw_size).unbind(1)
        keypoints = torch.stack([
            x_coords.cpu() * raw_w.unsqueeze(1) / width,
            y_coords.cpu() * raw_h.unsqueeze(1) / height,
        ], dim=-1)
        
        return keypoints.tolist(), scores.cpu().tolist()
    
    def __call__(self, outputs, inputs):
        keypoints, scores = self.heatmap2points(outputs, inputs['raw_size'])

        return keypoints, scores


def get_ball_postprocessor(args):
    postprocessor = BallPostprocessor()
    return postprocessor