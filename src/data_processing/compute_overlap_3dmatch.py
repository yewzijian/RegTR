"""Precomputes the overlap regions for 3DMatch dataset,
used for computing the losses in RegTR.
"""
import argparse
import os
import pickle
import sys
sys.path.append(os.getcwd())

import h5py
import numpy as np
import torch
from tqdm import tqdm

from utils.pointcloud import compute_overlap
from utils.se3_numpy import se3_transform, se3_init

parser = argparse.ArgumentParser()
# General
parser.add_argument('--base_dir', type=str, default='../data/indoor',
                    help='Path to 3DMatch raw data (Predator format)')
parser.add_argument('--overlap_radius', type=float, default=0.0375,
                    help='Overlap region will be sampled to this voxel size')
opt = parser.parse_args()


def process(phase):

    with open(f'datasets/3dmatch/{phase}_info.pkl', 'rb') as fid:
        infos = pickle.load(fid)

    out_file = os.path.join(opt.base_dir, f'{phase}_pairs-overlapmask.h5')
    print(f'Processing {phase}, output: {out_file}...')
    h5_fid = h5py.File(out_file, 'w')

    num_pairs = len(infos['src'])
    for item in tqdm(range(num_pairs)):
        src_path = infos['src'][item]
        tgt_path = infos['tgt'][item]
        pose = se3_init(infos['rot'][item], infos['trans'][item])  # transforms src to tgt

        src_xyz = torch.load(os.path.join(opt.base_dir, src_path))
        tgt_xyz = torch.load(os.path.join(opt.base_dir, tgt_path))

        src_mask, tgt_mask, src_tgt_corr = compute_overlap(
            se3_transform(pose, src_xyz),
            tgt_xyz,
            opt.overlap_radius,
        )

        h5_fid.create_dataset(f'/pair_{item:06d}/src_mask', data=src_mask)
        h5_fid.create_dataset(f'/pair_{item:06d}/tgt_mask', data=tgt_mask)
        h5_fid.create_dataset(f'/pair_{item:06d}/src_tgt_corr', data=src_tgt_corr)


if __name__ == '__main__':
    process('train')
    process('val')
    process('test_3DMatch')
    process('test_3DLoMatch')