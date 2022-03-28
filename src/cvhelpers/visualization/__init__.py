"""
Simple visualization tools
"""
from typing import List, Union

from matplotlib.pyplot import cm as colormap
import numpy as np
import open3d as o3d

from .visualizer import Visualizer
from .objects import *


def plotxyz(xyz: np.ndarray, colors: np.ndarray = None, axis_len: float = 0.0,
            pt_size: float = 1.0):
    """Plot point cloud

    Args:
        xyz: Point cloud of size (N,3)
        colors: Optional colors (N,3) or (3,). If not provided will be plotted in green
        axis_len: If a positive value is provided, will also plot the xyz axis
        pt_size: Size of points
    """
    vis = Visualizer()
    obj = create_point_cloud(xyz, colors, pt_size=pt_size)
    vis.add_object(obj)

    if axis_len > 0:
        vis.add_object(create_axes(axis_len))
    vis.start()


def plotxyz_multiple(xyz_list: List[Union[np.ndarray, o3d.geometry.PointCloud]], axis_len: float = 0.0,
                     pt_size: float = 1.0):
    """Visualize multiple point clouds in different colors

    Args:
        xyz_list: List of Nx3 point clouds
        axis_len: If a positive value is provided, will also plot the xyz axis
        pt_size: Size of points
    """

    vis = Visualizer()
    colors = colormap.get_cmap('Set2')

    for i in range(len(xyz_list)):
        if isinstance(xyz_list[i], o3d.geometry.PointCloud):
            xyz = np.asarray(xyz_list[i].points)
        else:
            xyz = xyz_list[i]

        color = (np.array(colors(i % colors.N)[:3]) * 255).astype(np.uint8)
        obj = create_point_cloud(xyz, colors=color, pt_size=pt_size)
        vis.add_object(obj)

    if axis_len > 0:
        vis.add_object(create_axes(axis_len))

    vis.start()


def plotxyz_mask(xyz: np.ndarray, mask: np.ndarray, axis_len: float = 0.0,
                 pt_size: float = 1.0):
    """Plot point cloud

    Args:
        xyz: Point cloud of size (N,3)
        mask: values True will be plotted as green, otherwise red.
        axis_len: If a positive value is provided, will also plot the xyz axis
        pt_size: Size of points
    """
    vis = Visualizer()
    colors = np.zeros((xyz.shape[0], 3), dtype=np.uint8)

    colors[mask, :] = np.array([[0, 255, 0]])
    colors[~mask, :] = np.array([[255, 0, 0]])

    obj = create_point_cloud(xyz, colors, pt_size=pt_size)
    vis.add_object(obj)

    if axis_len > 0:
        vis.add_object(vis.create_axes(axis_len))

    vis.start()
