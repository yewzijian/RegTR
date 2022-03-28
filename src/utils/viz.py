import numpy as np
import torch
from matplotlib.pyplot import cm as colormap

import cvhelpers.visualization as cvv
import cvhelpers.colors as colors
from cvhelpers.torch_helpers import to_numpy
from utils.se3_torch import se3_transform


def visualize_registration(src_xyz, tgt_xyz, correspondences,
                           correspondence_conf=None,
                           pose_gt=None, pose_pred=None):
    """Visualize registration, shown as a 2x3 grid:

    -------------
    | 0 | 1 | 2 |
    -------------
    | 3 | 4 | 5 |
    -------------

    0: Source point cloud with source keypoints
    1: Source and target point clouds, with lines indicating source keypoints to
       their transformed locations
    2: Source and target point clouds under groundtruth alignment (without clutter)
    3: Target point cloud with predicted transformed source keypoints
    4: Source and target point clouds under groundtruth alignment, with
       source keypoints and predited transformed coordinates, and a lines joining
       them (shorter lines means more accurate predictions)
    5: Source and target point clouds under predicted alignment (without clutter)

    Created 22 Oct 2021
    """

    if pose_gt is None:
        src_xyz_warped = src_xyz
        src_corr_warped = correspondences[:, :3]
    else:
        src_xyz_warped = se3_transform(pose_gt, src_xyz)
        src_corr_warped = se3_transform(pose_gt, correspondences[:, :3])

    vis = cvv.Visualizer(num_renderers=6, win_size=(1850, 1200))

    if correspondence_conf is None:
        src_kp_color = (255, 128, 128)
        tgt_kp_color = (128, 255, 128)
    else:
        conf = to_numpy(correspondence_conf)
        src_color_mapper = colormap.ScalarMappable(norm=None, cmap=colormap.get_cmap('autumn'))
        src_kp_color = (src_color_mapper.to_rgba(conf)[:, :3] * 255).astype(np.uint8)
        tgt_color_mapper = colormap.ScalarMappable(norm=None, cmap=colormap.get_cmap('summer'))
        tgt_kp_color = (tgt_color_mapper.to_rgba(conf)[:, :3] * 255).astype(np.uint8)
    # Show points on source
    vis.add_object(
        cvv.create_point_cloud(src_xyz_warped, colors=colors.RED),
        renderer_idx=0,
    )
    vis.add_object(
        cvv.create_point_cloud(src_corr_warped, colors=src_kp_color, pt_size=4),
        renderer_idx=0,
    )

    # Show points on target
    vis.add_object(
        cvv.create_point_cloud(tgt_xyz, colors=colors.GREEN),
        renderer_idx=3,
    )
    vis.add_object(
        cvv.create_point_cloud(correspondences[:, 3:], colors=tgt_kp_color, pt_size=4),
        renderer_idx=3,
    )

    # Show correspondences with lines joining the two
    vis.add_object(
        cvv.create_point_cloud(src_xyz, colors=colors.RED),
        renderer_idx=1,
    )
    vis.add_object(
        cvv.create_point_cloud(tgt_xyz, colors=colors.GREEN),
        renderer_idx=1,
    )
    vis.add_object(
        cvv.create_lines(correspondences),
        renderer_idx=1
    )

    # Show overlap using groundtruth pose
    vis.add_object(
        cvv.create_point_cloud(src_xyz_warped, colors=colors.RED),
        renderer_idx=4,
    )
    vis.add_object(
        cvv.create_point_cloud(tgt_xyz, colors=colors.GREEN),
        renderer_idx=4,
    )
    vis.add_object(
        cvv.create_point_cloud(src_corr_warped, colors=src_kp_color, pt_size=4),
        renderer_idx=4
    )
    vis.add_object(
        cvv.create_point_cloud(correspondences[:, 3:], colors=tgt_kp_color, pt_size=4),
        renderer_idx=4
    )
    vis.add_object(
        cvv.create_lines(torch.cat([src_corr_warped, correspondences[:, 3:]], dim=1)),
        renderer_idx=4
    )

    # Show groundtruth (without clutter)
    vis.add_object(
        cvv.create_point_cloud(src_xyz_warped, colors=colors.RED),
        renderer_idx=2,
    )
    vis.add_object(
        cvv.create_point_cloud(tgt_xyz, colors=colors.GREEN),
        renderer_idx=2,
    )

    # Show predicted pose
    if pose_pred is not None:
        vis.add_object(
            cvv.create_point_cloud(se3_transform(pose_pred, src_xyz), colors=colors.RED),
            renderer_idx=5,
        )
        vis.add_object(
            cvv.create_point_cloud(tgt_xyz, colors=colors.GREEN),
            renderer_idx=5,
        )

    # Render loop
    vis.reset_camera()
    vis.start()
