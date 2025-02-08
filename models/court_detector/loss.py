"""Loss for court detector."""
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

    def calc_pixel_dist(self, keypoints, scores, targets):
        """ Measure keypoint pixel error(l2 norm).

        keypoints: batch, channel, 2
        """
        bs = len(keypoints)
        # make gt array to use matrix operation
        preds = np.array(keypoints)
        gts = np.zeros((bs, 14, 2))
        foreground = np.zeros((bs, 14))

        for b in range(bs):
            kps = targets['raw_keypoints'][b]
            labels = list(map(int, targets['raw_labels'][b]))
            gts[b][labels] = kps
            foreground[b][labels] = 1
        l2_norm = np.linalg.norm(preds - gts, axis=2) * foreground
        return l2_norm.sum() / foreground.sum()
    
    def calc_confusion(self, confusion, keypoints, scores, targets, score_threshold=0.5, distance_threshold=7):
        for gt, label, pred, score in zip(targets['raw_keypoints'], targets['raw_labels'], keypoints, scores):
            dic = {int(label): kp for label, kp in zip(label, gt)}
            for i, (pred_pt, score_pt) in enumerate(zip(pred, score)):
                # true
                if i in dic:
                    gt_pt = dic[i]
                    dist =  ((pred_pt[0] - gt_pt[0])**2 + (pred_pt[1] - gt_pt[1])**2)**0.5
                    if score_pt > score_threshold and dist < distance_threshold:
                        confusion['tp'] += 1
                    else:
                        confusion['fn'] += 1
                else:
                    if score_pt > score_threshold:
                        confusion['fp'] += 1
                    else:
                        confusion['tn'] += 1

def get_criterion(args):
    criterion = Criterion(args.focal_alpha, args.focal_beta)
    return criterion    