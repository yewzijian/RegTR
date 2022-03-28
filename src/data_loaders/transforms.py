"""
Data augmentations for use during training.
Note that the operations are in-place
"""
import random

import numpy as np
from cvhelpers.lie.numpy import SE3
from scipy.spatial.transform import Rotation
import torch

from utils.se3_torch import se3_inv, se3_init, se3_cat, se3_transform


class RigidPerturb:
    """Applies a random rigid transform to either the source or target point
    cloud.

    Args:
        perturb_mode: Either 'none', 'small', or 'large'. 'large' is the same
          as in Predator. 'small' just performs a slight perturbation
    """
    def __init__(self, perturb_mode='small'):
        assert perturb_mode in ['none', 'small', 'large']
        self.perturb_mode = perturb_mode

    @staticmethod
    def _sample_pose_large():
        euler_ab = np.random.rand(3) * np.pi * 2  # anglez, angley, anglex
        rot_ab = Rotation.from_euler('zyx', euler_ab).as_matrix()
        perturb = np.concatenate([rot_ab, np.zeros((3, 1))], axis=1)
        return torch.from_numpy(perturb).float()

    @staticmethod
    def _sample_pose_small(std=0.1):
        perturb = SE3.sample_small(std=std).as_matrix()
        return torch.from_numpy(perturb).float()

    def __call__(self, data):
        if self.perturb_mode == 'none':
            return data

        perturb = self._sample_pose_small() if self.perturb_mode == 'small' else \
            self._sample_pose_large()

        perturb_source = random.random() > 0.5  # whether to perturb source or target

        if self.perturb_mode == 'small':
            # Center perturbation around the point centroid (otherwise there's a large
            # induced translation as rotation is centered around origin)
            centroid = torch.mean(data['src_xyz'], dim=0).unsqueeze(1) if perturb_source else \
                torch.mean(data['tgt_xyz'], dim=0).unsqueeze(1)
            center_transform = se3_init(rot=None, trans=-centroid)
            perturb = se3_cat(se3_cat(se3_inv(center_transform), perturb), center_transform)

        if perturb_source:
            data['pose'] = se3_cat(data['pose'], se3_inv(perturb))
            data['src_xyz'] = se3_transform(perturb, data['src_xyz'])
            if 'corr_xyz' in data:
                data['corr_xyz'][:, :3] = se3_transform(perturb, data['corr_xyz'][:, :3])
            if 'corr_xyz_ds' in data:
                data['corr_xyz_ds'][:, :3] = se3_transform(perturb, data['corr_xyz_ds'][:, :3])

        else:
            data['pose'] = se3_cat(perturb, data['pose'])
            data['tgt_xyz'] = se3_transform(perturb, data['tgt_xyz'])
            if 'corr_xyz' in data:
                data['corr_xyz'][:, 3:] = se3_transform(perturb, data['corr_xyz'][:, 3:])
            if 'corr_xyz_ds' in data:
                data['corr_xyz_ds'][:, 3:] = se3_transform(perturb, data['corr_xyz_ds'][:, 3:])

        return data


class Jitter:
    """Jitter the position by a small amount

    Args:
        scale: Controls the amount to jitter. Noise will be sampled from
           a gaussian distribution with standard deviation given by scale,
           independently for each axis
    """
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def __call__(self, data):

        for cloud in ['src_xyz', 'tgt_xyz']:
            noise = torch.randn(data[cloud].shape) * self.scale
            data[cloud] = data[cloud] + noise
        return data


class ShufflePoints:
    """Shuffle the points
    """
    def __init__(self, max_pts=30000, shuffle=True):
        super().__init__()
        self.max_pts = max_pts
        self.shuffle = shuffle

    def __call__(self, data):

        if self.shuffle:
            src_idx = np.random.permutation(data['src_xyz'].shape[0])[:self.max_pts]
            tgt_idx = np.random.permutation(data['tgt_xyz'].shape[0])[:self.max_pts]
        else:
            src_idx = np.arange(min(data['src_xyz'].shape[0], self.max_pts))
            tgt_idx = np.arange(min(data['tgt_xyz'].shape[0], self.max_pts))

        # Compute reverse indices
        if 'correspondences' in data:
            src_idx_rev = np.full(data['src_xyz'].shape[0], -1)
            src_idx_rev[src_idx] = np.arange(src_idx.shape[0])
            tgt_idx_rev = np.full(data['tgt_xyz'].shape[0], -1)
            tgt_idx_rev[tgt_idx] = np.arange(tgt_idx.shape[0])
            src_idx_rev = torch.from_numpy(src_idx_rev)
            tgt_idx_rev = torch.from_numpy(tgt_idx_rev)
            correspondences = torch.stack(
                [src_idx_rev[data['correspondences'][0]], tgt_idx_rev[data['correspondences'][1]]])
            data['correspondences'] = correspondences[:, torch.all(correspondences >= 0, dim=0)]

        data['src_xyz'] = data['src_xyz'][src_idx, :]
        data['src_overlap'] = data['src_overlap'][src_idx]
        data['tgt_xyz'] = data['tgt_xyz'][tgt_idx, :]
        data['tgt_overlap'] = data['tgt_overlap'][tgt_idx]

        return data


class RandomSwap:
    """Swaps the source and target point cloud with a 50% chance"""
    def __init__(self):
        pass

    def __call__(self, data):
        if random.random() > 0.5:
            data['src_xyz'], data['tgt_xyz'] = data['tgt_xyz'], data['src_xyz']
            data['src_path'], data['tgt_path'] = data['tgt_path'], data['src_path']
            data['src_overlap'], data['tgt_overlap'] = data['tgt_overlap'], data['src_overlap']
            if 'correspondences' in data:
                data['correspondences'] = torch.stack([data['correspondences'][1], data['correspondences'][0]])
            if 'corr_xyz' in data:
                data['corr_xyz'] = torch.cat([data['corr_xyz'][:, 3:], data['corr_xyz'][:, :3]], dim=1)
            if 'corr_xyz_ds' in data:
                data['corr_xyz_ds'] = torch.cat([data['corr_xyz_ds'][:, 3:], data['corr_xyz_ds'][:, :3]], dim=1)
            data['pose'] = se3_inv(data['pose'])
        return data
