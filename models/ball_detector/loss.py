"""Loss for ball detector."""
import torch
import torch.nn as nn
import numpy as np


class FocalLoss(nn.Module):
    """Focal Loss."""

    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        """Init."""
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt, mask, eps=1e-7):
        """Forward."""
        neg_inds = torch.ones_like(mask) - mask

        neg_weights = torch.pow(1 - gt, self.beta)

        pos_loss = torch.log(pred + eps) * torch.pow(1 - pred, self.alpha) * mask
        neg_loss = torch.log(1 - pred + eps) * torch.pow(pred, self.alpha) * neg_weights * neg_inds

        pos_loss = pos_loss.sum(dim=[1, 2, 3]).mean()
        neg_loss = neg_loss.sum(dim=[1, 2, 3]).mean()
        num_pos = mask.sum().clamp(1, 1e30)
        return -(pos_loss + neg_loss) / num_pos


class Criterion(nn.Module):
    """Criterion."""

    def __init__(self, alpha, beta):
        """Init."""
        super().__init__()
        self.criterion = FocalLoss(alpha, beta)

    def forward(self, outputs, targets, device):
        """Loss calculation.

        outputs: batch_size, intermediates, out_ch, h, w
        targets['heatmap']: batch_size, out_ch, h, w
        mask: batch_size, out_ch, h, w
        """
        heatmap = targets['heatmap'].to(device)
        mask = targets['mask'].to(device)
        
        losses = []
        for output in outputs.unbind(1):
            loss = self.criterion(output.sigmoid(), heatmap, mask)
            losses.append(loss)

        return sum(losses) / len(losses)

    def calc_pixel_dist(self, keypoints, scores, targets, threshold=0.0):
        """ Measure keypoint pixel error(l2 norm).

        keypoints: batch, channel, 2
        """
        bs = len(keypoints)
        # make gt array to use matrix operation
        # preds = np.array(keypoints)
        preds = np.zeros((bs, 1, 2))
        for b in range(bs):
            for kp, score in zip(keypoints[b], scores[b]):
                if score > threshold:
                    preds[b][:] = kp
        
        gts = np.zeros((bs, 1, 2))
        foreground = np.zeros((bs))
        for b in range(bs):
            kps = targets['raw_keypoints'][b]
            if kps:
                gts[b][:] = kps
                foreground[b] = 1
        l2_norm = np.linalg.norm(preds - gts, axis=2) * foreground
        return l2_norm.sum() / foreground.sum()

    def calc_confusion(self, confusion, keypoints, scores, targets, score_threshold=0.5, distance_threshold=7):
        for gt, pred, score in zip(targets['raw_keypoints'], keypoints, scores):
            if gt:
                dist = np.linalg.norm(np.array(pred)-np.array(gt), ord=2)
                if score[0] > score_threshold and dist < distance_threshold:
                    confusion['tp'] += 1
                else:
                    confusion['fn'] += 1
            else:
                if score[0] > score_threshold:
                    confusion['fp'] += 1
                else:
                    confusion['tn'] += 1

def get_criterion(args):
    criterion = Criterion(args.focal_alpha, args.focal_beta)
    return criterion