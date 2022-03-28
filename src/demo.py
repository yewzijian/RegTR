"""Barebones code demonstrating REGTR's registration. We provide 2 demo
instances for each dataset

Simply download the pretrained weights from the project webpage, then run:
    python demo.py EXAMPLE_IDX
where EXAMPLE_IDX can be a number between 0-5 (defined at line 25)

The registration results will be shown in a 3D visualizer.
"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from easydict import EasyDict
from matplotlib.pyplot import cm as colormap

import cvhelpers.visualization as cvv
import cvhelpers.colors as colors
from cvhelpers.torch_helpers import to_numpy
from models.regtr import RegTR
from utils.misc import load_config
from utils.se3_numpy import se3_transform

_examples = [
    # 3DMatch examples
    # 0
    ('../trained_models/3dmatch/ckpt/model-best.pth',
     '../data/indoor/test/7-scenes-redkitchen/cloud_bin_0.pth',
     '../data/indoor/test/7-scenes-redkitchen/cloud_bin_5.pth'),
    # 1
    ('../trained_models/3dmatch/ckpt/model-best.pth',
     '../data/indoor/test/sun3d-hotel_umd-maryland_hotel3/cloud_bin_8.pth',
     '../data/indoor/test/sun3d-hotel_umd-maryland_hotel3/cloud_bin_15.pth'),
    # 2
    ('../trained_models/3dmatch/ckpt/model-best.pth',
     '../data/indoor/test/sun3d-home_at-home_at_scan1_2013_jan_1/cloud_bin_38.pth',
     '../data/indoor/test/sun3d-home_at-home_at_scan1_2013_jan_1/cloud_bin_41.pth'),
    # ModelNet examples
    # 3
    ('../trained_models/modelnet/ckpt/model-best.pth',
     '../data/modelnet_demo_data/modelnet_test_2_0.ply',
     '../data/modelnet_demo_data/modelnet_test_2_1.ply'),
    # 4
    ('../trained_models/modelnet/ckpt/model-best.pth',
     '../data/modelnet_demo_data/modelnet_test_630_0.ply',
     '../data/modelnet_demo_data/modelnet_test_630_1.ply'),
]

parser = argparse.ArgumentParser()
parser.add_argument('--example', type=int, default=0,
                    help=f'Example pair to run (between 0 and {len(_examples) - 1})')
parser.add_argument('--threshold', type=float, default=0.5,
                    help='Controls viusalization of keypoints outside overlap region.')
opt = parser.parse_args()


def visualize_result(src_xyz: np.ndarray, tgt_xyz: np.ndarray,
                     src_kp: np.ndarray, src2tgt: np.ndarray,
                     src_overlap: np.ndarray,
                     pose: np.ndarray,
                     threshold: float = 0.5):
    """Visualizes the registration result:
       - Top-left: Source point cloud and keypoints
       - Top-right: Target point cloud and predicted corresponding kp positions
       - Bottom-left: Source and target point clouds before registration
       - Bottom-right: Source and target point clouds after registration

    Press 'q' to exit.

    Args:
        src_xyz: Source point cloud (M x 3)
        tgt_xyz: Target point cloud (N x 3)
        src_kp: Source keypoints (M' x 3)
        src2tgt: Corresponding positions of src_kp in target (M' x 3)
        src_overlap: Predicted probability the point lies in the overlapping region
        pose: Estimated rigid transform
        threshold: For clarity, we only show predicted overlapping points (prob > 0.5).
                   Set to 0 to show all keypoints, and a larger number to show
                   only points strictly within the overlap region.
    """

    large_pt_size = 4
    color_mapper = colormap.ScalarMappable(norm=None, cmap=colormap.get_cmap('coolwarm'))
    overlap_colors = (color_mapper.to_rgba(src_overlap[:, 0])[:, :3] * 255).astype(np.uint8)
    m = src_overlap[:, 0] > threshold

    vis = cvv.Visualizer(
        win_size=(1600, 1000),
        num_renderers=4)

    vis.add_object(
        cvv.create_point_cloud(src_xyz, colors=colors.RED),
        renderer_idx=0
    )
    vis.add_object(
        cvv.create_point_cloud(src_kp[m, :], colors=overlap_colors[m, :], pt_size=large_pt_size),
        renderer_idx=0
    )

    vis.add_object(
        cvv.create_point_cloud(tgt_xyz, colors=colors.GREEN),
        renderer_idx=1
    )
    vis.add_object(
        cvv.create_point_cloud(src2tgt[m, :], colors=overlap_colors[m, :], pt_size=large_pt_size),
        renderer_idx=1
    )

    # Before registration
    vis.add_object(
        cvv.create_point_cloud(src_xyz, colors=colors.RED),
        renderer_idx=2
    )
    vis.add_object(
        cvv.create_point_cloud(tgt_xyz, colors=colors.GREEN),
        renderer_idx=2
    )

    # After registration
    vis.add_object(
        cvv.create_point_cloud(se3_transform(pose, src_xyz), colors=colors.RED),
        renderer_idx=3
    )
    vis.add_object(
        cvv.create_point_cloud(tgt_xyz, colors=colors.GREEN),
        renderer_idx=3
    )

    vis.set_titles(['Source point cloud (with keypoints)',
                    'Target point cloud (with predicted source keypoint positions)',
                    'Before Registration',
                    'After Registration'])

    vis.reset_camera()
    vis.start()


def load_point_cloud(fname):
    if fname.endswith('.pth'):
        data = torch.load(fname)
    elif fname.endswith('.ply'):
        pcd = o3d.io.read_point_cloud(fname)
        data = np.asarray(pcd.points)
    elif fname.endswith('.bin'):
        data = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
    else:
        raise AssertionError('Cannot recognize point cloud format')

    return data[:, :3]  # ignore reflectance, or other features if any


def main():
    # Retrieves the model and point cloud paths
    ckpt_path, src_path, tgt_path = _examples[opt.example]

    # Load configuration file
    cfg = EasyDict(load_config(Path(ckpt_path).parents[1] / 'config.yaml'))
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Instantiate model and restore weights
    model = RegTR(cfg).to(device)
    state = torch.load(ckpt_path)
    model.load_state_dict(state['state_dict'])

    # Loads point cloud data: Each is represented as a Nx3 numpy array
    src_xyz = load_point_cloud(src_path)
    tgt_xyz = load_point_cloud(tgt_path)

    if 'crop_radius' in cfg:
        # Crops the point cloud if necessary (set in the config file)
        crop_radius = cfg['crop_radius']
        src_xyz = src_xyz[np.linalg.norm(src_xyz, axis=1) < crop_radius, :]
        tgt_xyz = tgt_xyz[np.linalg.norm(tgt_xyz, axis=1) < crop_radius, :]

    # Feeds the data into the model
    data_batch = {
        'src_xyz': [torch.from_numpy(src_xyz).float().to(device)],
        'tgt_xyz': [torch.from_numpy(tgt_xyz).float().to(device)]
    }
    outputs = model(data_batch)

    # Visualize the results
    b = 0
    pose = to_numpy(outputs['pose'][-1, b])
    src_kp = to_numpy(outputs['src_kp'][b])
    src2tgt = to_numpy(outputs['src_kp_warped'][b][-1])  # pred. corresponding locations of src_kp
    overlap_score = to_numpy(torch.sigmoid(outputs['src_overlap'][b][-1]))

    visualize_result(src_xyz, tgt_xyz, src_kp, src2tgt, overlap_score, pose,
                     threshold=opt.threshold)


if __name__ == '__main__':
    main()