import torch
import torch.nn as nn

from utils.se3_torch import se3_transform_list

_EPS = 1e-6


class CorrCriterion(nn.Module):
    """Correspondence Loss.
    """
    def __init__(self, metric='mae'):
        super().__init__()
        assert metric in ['mse', 'mae']

        self.metric = metric

    def forward(self, kp_before, kp_warped_pred, pose_gt, overlap_weights=None):

        losses = {}
        B = pose_gt.shape[0]

        kp_warped_gt = se3_transform_list(pose_gt, kp_before)
        corr_err = torch.cat(kp_warped_pred, dim=0) - torch.cat(kp_warped_gt, dim=0)

        if self.metric == 'mae':
            corr_err = torch.sum(torch.abs(corr_err), dim=-1)
        elif self.metric == 'mse':
            corr_err = torch.sum(torch.square(corr_err), dim=-1)
        else:
            raise NotImplementedError

        if overlap_weights is not None:
            overlap_weights = torch.cat(overlap_weights)
            mean_err = torch.sum(overlap_weights * corr_err) / torch.clamp_min(torch.sum(overlap_weights), _EPS)
        else:
            mean_err = torch.mean(corr_err, dim=1)

        return mean_err

