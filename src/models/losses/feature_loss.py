import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-6


def cdist(a, b, metric='euclidean'):
    """Similar to scipy.spatial's cdist, but symbolic.
    The currently supported metrics can be listed as `cdist.supported_metrics` and are:
        - 'euclidean', although with a fudge-factor epsilon.
        - 'sqeuclidean', the squared euclidean.
        - 'cityblock', the manhattan or L1 distance.
    Args:
        a: The left-hand side, shaped ([*,] F, B1).  <- Not that dimension ordering is different from torch.cdist
        b: The right-hand side, shaped ([*,], F, B2).
        metric (string): Which distance metric to use, see notes.
    Returns:
        The matrix of all pairwise distances between all vectors in `a` and in
        `b`, will be of shape (B1, B2).
    Note:
        When a square root is taken (such as in the Euclidean case), a small
        epsilon is added because the gradient of the square-root at zero is
        undefined. Thus, it will never return exact zero in these cases.

    Taken from Predator source code, which was modified from D3Feat.
    """
    if metric == 'sqeuclidean':
        diffs = a[..., :, None] - b[..., None, :]
        return torch.sum(diffs ** 2, dim=-3)
    elif metric == 'euclidean':
        diffs = a[..., :, None] - b[..., None, :]
        return torch.sqrt(torch.sum(diffs ** 2, dim=-3) + 1e-12)
    elif metric == 'cityblock':
        diffs = a[..., :, None] - b[..., None, :]
        return torch.sum(torch.abs(diffs), dim=-3)
    elif metric == 'cosine':
        numer = a.transpose(-1, -2) @ b
        denom = torch.clamp_min(
            torch.norm(a, dim=-2)[..., :, None] * torch.norm(b, dim=-2)[..., None, :],
            1e-8)
        dist = 1 - numer / denom
        return dist
    else:
        raise NotImplementedError(
            'The following metric is not implemented by `cdist` yet: {}'.format(metric))


class CircleLoss(nn.Module):
    """Circle triplet loss on feature descriptors

    Modified from source codes of:
     - D3Feat https://github.com/XuyangBai/D3Feat.pytorch/,
     - Predator https://github.com/overlappredator/OverlapPredator
    """
    def __init__(self, dist_type='cosine', log_scale=10, r_p=0.125, r_n=0.25, pos_margin=0.1, neg_margin=1.4):
        """

        Args:
            dist_type: Distance type for comparing features
            log_scale:
            r_p: Radius where points < r_p away will be considered matching
            r_n: Radius where points > r_p away will be considered non-matching
            pos_margin: Circle loss margin for better similarity separation (pos)
            neg_margin: Circle loss margin for better similarity separation (neg)
        """
        super().__init__()
        self.log_scale = log_scale
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.pos_optimal = pos_margin
        self.neg_optimal = neg_margin
        self.dist_type = dist_type

        self.r_p = r_p
        self.r_n = r_n
        self.n_sample = 256  # Number of correspondences to sample for each pair
        self.logger = logging.getLogger(__name__)

    def get_circle_loss(self, coords_dist, feats_dist):
        """Computes circle loss given feature distances
        Modified from implementations from Predator and D3Feat source codes

        Args:
            coords_dist: (B, N_src, N_tgt)
            feats_dist: (B, N_src, N_tgt)

        Returns:

        """
        pos_mask = coords_dist < self.r_p
        neg_mask = coords_dist > self.r_n

        # get points that have both positive and negative pairs
        row_sel = ((pos_mask.sum(-1) > 0) * (neg_mask.sum(-1) > 0)).detach()
        col_sel = ((pos_mask.sum(-2) > 0) * (neg_mask.sum(-2) > 0)).detach()

        pos = feats_dist - 1e5 * (~pos_mask).float()  # mask the non-positive: exp(-1e5) ~ 0
        pos_weight = torch.clamp_min(pos - self.pos_optimal, min=0).detach()
        lse_positive_row = torch.logsumexp(self.log_scale * (pos - self.pos_margin) * pos_weight,
                                           dim=-1)
        lse_positive_col = torch.logsumexp(self.log_scale * (pos - self.pos_margin) * pos_weight,
                                           dim=-2)

        neg = feats_dist + 1e5 * (~neg_mask).float()
        neg_weight = torch.clamp_min(self.neg_optimal - neg, min=0).detach()
        lse_negative_row = torch.logsumexp(self.log_scale * (self.neg_margin - neg) * neg_weight,
                                           dim=-1)
        lse_negative_col = torch.logsumexp(self.log_scale * (self.neg_margin - neg) * neg_weight,
                                           dim=-2)

        loss_row = F.softplus(lse_positive_row + lse_negative_row) / self.log_scale
        loss_col = F.softplus(lse_positive_col + lse_negative_col) / self.log_scale
        if row_sel.sum() == 0 or col_sel.sum() == 0:
            self.logger.warning('No valid pairs: row_sum={}, col_sum={}.'
                                .format(row_sel.sum(), col_sel.sum()))

        circle_loss = (loss_row[row_sel].mean() + loss_col[col_sel].mean()) / 2
        return circle_loss

    def forward(self, anchor_feat, positive_feat, anchor_xyz, positive_xyz,
                anchor_batch, positive_batch):

        assert anchor_feat.shape[0] == anchor_xyz.shape[0] == anchor_batch.shape[0]
        assert positive_feat.shape[0] == positive_xyz.shape[0] == positive_batch.shape[0]

        B = anchor_batch.max() + 1

        # Compute groundtruth correspondences on the fly
        correspondences = radius_search(positive_xyz, anchor_xyz, self.r_p - 0.001,
                                        positive_batch.type(torch.int64), anchor_batch.type(torch.int64))
        corr_batch = anchor_batch[correspondences[0]]

        sel_idx_all = []
        for b in range(B):
            b_idx = torch.nonzero(corr_batch == b)[:, 0]
            sel_idx = np.random.choice(b_idx.cpu().numpy(), self.n_sample,
                                       replace=len(b_idx) < self.n_sample)
            sel_idx_all.append(sel_idx)
        sel_idx_all = np.concatenate(sel_idx_all)

        correspondences = correspondences[:, sel_idx_all]
        src_feats = anchor_feat[correspondences[0]].view(B, self.n_sample, -1)
        tgt_feats = positive_feat[correspondences[1]].view(B, self.n_sample, -1)
        src_xyz = anchor_xyz[correspondences[0]].view(B, self.n_sample, -1)
        tgt_xyz = positive_xyz[correspondences[1]].view(B, self.n_sample, -1)

        coords_dist = torch.cdist(src_xyz, tgt_xyz)
        # feats_dist = torch.cdist(src_feats, tgt_feats)
        feats_dist = cdist(src_feats.transpose(-1, -2), tgt_feats.transpose(-1, -2), metric=self.dist_type)

        circle_loss = self.get_circle_loss(coords_dist, feats_dist)

        return circle_loss


class CircleLossFull(nn.Module):
    """Circle triplet loss on feature descriptors. This version uses all
    descriptors instead of sampling

    Modified from source codes of:
     - D3Feat https://github.com/XuyangBai/D3Feat.pytorch/,
     - Predator https://github.com/overlappredator/OverlapPredator
    """
    def __init__(self, dist_type='cosine', log_scale=10, r_p=0.125, r_n=0.25, pos_margin=0.1, neg_margin=1.4):
        """

        Args:
            dist_type: Distance type for comparing features
            log_scale:
            r_p: Radius where points < r_p away will be considered matching
            r_n: Radius where points > r_p away will be considered non-matching
            pos_margin: Circle loss margin for better similarity separation (pos)
            neg_margin: Circle loss margin for better similarity separation (neg)
        """
        super().__init__()
        self.log_scale = log_scale
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.pos_optimal = pos_margin
        self.neg_optimal = neg_margin
        self.dist_type = dist_type

        self.r_p = r_p
        self.r_n = r_n
        self.logger = logging.getLogger(__name__)

    def get_circle_loss(self, coords_dist, feats_dist):
        """Computes circle loss given feature distances
        Modified from implementations from Predator and D3Feat source codes

        Args:
            coords_dist: (*, N_src, N_tgt)
            feats_dist: (*, N_src, N_tgt)

        Returns:

        """
        pos_mask = coords_dist < self.r_p
        neg_mask = coords_dist > self.r_n

        # get points that have both positive and negative pairs
        row_sel = ((pos_mask.sum(-1) > 0) * (neg_mask.sum(-1) > 0)).detach()
        col_sel = ((pos_mask.sum(-2) > 0) * (neg_mask.sum(-2) > 0)).detach()

        pos = feats_dist - 1e5 * (~pos_mask).float()  # mask the non-positive: exp(-1e5) ~ 0
        pos_weight = torch.clamp_min(pos - self.pos_optimal, min=0).detach()
        lse_positive_row = torch.logsumexp(self.log_scale * (pos - self.pos_margin) * pos_weight,
                                           dim=-1)
        lse_positive_col = torch.logsumexp(self.log_scale * (pos - self.pos_margin) * pos_weight,
                                           dim=-2)

        neg = feats_dist + 1e5 * (~neg_mask).float()
        neg_weight = torch.clamp_min(self.neg_optimal - neg, min=0).detach()
        lse_negative_row = torch.logsumexp(self.log_scale * (self.neg_margin - neg) * neg_weight,
                                           dim=-1)
        lse_negative_col = torch.logsumexp(self.log_scale * (self.neg_margin - neg) * neg_weight,
                                           dim=-2)

        loss_row = F.softplus(lse_positive_row + lse_negative_row) / self.log_scale
        loss_col = F.softplus(lse_positive_col + lse_negative_col) / self.log_scale
        if row_sel.sum() == 0 or col_sel.sum() == 0:
            self.logger.warning('No valid pairs: row_sum={}, col_sum={}.'
                                .format(row_sel.sum(), col_sel.sum()))

        circle_loss = (loss_row[row_sel].mean() + loss_col[col_sel].mean()) / 2
        return circle_loss

    def forward(self, anchor_feat, positive_feat, anchor_xyz, positive_xyz):

        B = len(anchor_feat)
        assert all([anchor_feat[b].shape[0] == anchor_xyz[b].shape[0] for b in range(B)])
        assert all([positive_feat[b].shape[0] == positive_xyz[b].shape[0] for b in range(B)])

        circle_loss = 0
        for b in range(B):
            coords_dist = torch.cdist(anchor_xyz[b], positive_xyz[b])
            feats_dist = cdist(anchor_feat[b].transpose(-1, -2), positive_feat[b].transpose(-1, -2), metric=self.dist_type)
            circle_loss += self.get_circle_loss(coords_dist, feats_dist)
        return circle_loss / B


class InfoNCELossFull(nn.Module):
    """Computes InfoNCE loss
    """
    def __init__(self, d_embed, r_p, r_n):
        """
        Args:
            d_embed: Embedding dimension
            r_p: Positive radius (points nearer than r_p are matches)
            r_n: Negative radius (points nearer than r_p are not matches)
        """

        super().__init__()
        self.r_p = r_p
        self.r_n = r_n
        self.n_sample = 256
        self.W = torch.nn.Parameter(torch.zeros(d_embed, d_embed), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.W, std=0.1)

    def compute_infonce(self, anchor_feat, positive_feat, anchor_xyz, positive_xyz):
        """

        Args:
            anchor_feat: Shape ([B,] N_anc, D)
            positive_feat: Shape ([B,] N_pos, D)
            anchor_xyz: ([B,] N_anc, 3)
            positive_xyz: ([B,] N_pos, 3)

        Returns:
        """

        W_triu = torch.triu(self.W)
        W_symmetrical = W_triu + W_triu.T
        match_logits = torch.einsum('...ic,cd,...jd->...ij', anchor_feat, W_symmetrical, positive_feat)  # (..., N_anc, N_pos)

        with torch.no_grad():
            dist_keypts = torch.cdist(anchor_xyz, positive_xyz)

            dist1, idx1 = dist_keypts.topk(k=1, dim=-1, largest=False)  # Finds the positive (closest match)
            mask = dist1[..., 0] < self.r_p  # Only consider points with correspondences (..., N_anc)
            ignore = dist_keypts < self.r_n  # Ignore all the points within a certain boundary,
            ignore.scatter_(-1, idx1, 0)     # except the positive (..., N_anc, N_pos)

        match_logits[..., ignore] = -float('inf')

        loss = -torch.gather(match_logits, -1, idx1).squeeze(-1) + torch.logsumexp(match_logits, dim=-1)
        loss = torch.sum(loss[mask]) / torch.sum(mask)
        return loss

    def forward(self, src_feat, tgt_feat, src_xyz, tgt_xyz):
        """

        Args:
            src_feat: List(B) of source features (N_src, D)
            tgt_feat: List(B) of target features (N_tgt, D)
            src_xyz:  List(B) of source coordinates (N_src, 3)
            tgt_xyz: List(B) of target coordinates (N_tgt, 3)

        Returns:

        """

        B = len(src_feat)
        infonce_loss = [self.compute_infonce(src_feat[b], tgt_feat[b], src_xyz[b], tgt_xyz[b]) for b in range(B)]

        return torch.mean(torch.stack(infonce_loss))
