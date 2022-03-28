import numpy as np
import open3d as o3d


def to_o3d_pcd(xyz, colors=None, normals=None):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)

    return pcd


def to_o3d_feats(embedding: np.ndarray):
    """
    Convert tensor/array to open3d features
    embedding:  [N, D]
    """
    feats = o3d.pipelines.registration.Feature()
    feats.data = embedding.T
    return feats