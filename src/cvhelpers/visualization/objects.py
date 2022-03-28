"""Functions to create objects to add to the visualizer"""

import numpy as np
import torch

from .vtk_object import VTKObject


def _convert_torch_to_numpy(arr):
    """If arr is torch.Tensor, return the numpy equivalent, else return arr
    as it is"""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    return arr


def create_point_cloud(xyz: np.ndarray, colors=None, cmap=None, color_norm=None,
                       pt_size=1.0, alpha=1.0):
    """Create a point cloud with colors from a given NumPy array

    The NumPy array should have dimension Nx6 where the first three
    dimensions correspond to X, Y and Z and the last three dimensions
    correspond to R, G and B values (between 0 and 255)

    Returns: VTKObject() which encapulsates the point sources and actors
    """

    xyz = _convert_torch_to_numpy(xyz)

    obj = VTKObject()
    obj.CreateFromArray(xyz[:, :3])
    if colors is not None:
        obj.SetColors(colors, cmap, color_norm)
    if alpha < 1.0:
        obj.actor.GetProperty().SetOpacity(alpha)
    obj.actor.GetProperty().SetPointSize(pt_size)
    return obj


def create_hedgehog_actor(xyz, normals, scale=1.0):
    obj = VTKObject()
    obj.CreateFromArray(xyz)
    obj.AddNormals(normals)
    obj.SetupPipelineHedgeHog(scale)
    return obj


def create_axes(length):
    """Create coordinate system axes with specified length"""
    obj = VTKObject()
    obj.CreateAxes(length)
    return obj


def create_sphere(origin, r=1.0, color=None):
    """Create a sphere with given origin (x,y,z) and radius r"""

    origin = _convert_torch_to_numpy(origin)

    obj = VTKObject()
    obj.CreateSphere(origin, r, color)
    return obj


def create_cylinder(origin, r=1.0, h=1.0):
    """Create a cylinder with given origin (x,y,z), radius r and height h"""
    obj = VTKObject()
    obj.CreateCylinder(origin, r, h)
    return obj


def create_plane(normal=None, origin=None):
    """Create a plane (optionally with a given normal vector and origin)

    Note: SetActorScale can be used to scale the extent of the plane"""
    obj = VTKObject()
    obj.CreatePlane(normal, origin)
    return obj


def create_box(bounds):
    """Create a box witih the given bounds=[xmin,xmax,ymin,ymax,zmin,zmax]"""
    obj = VTKObject()
    obj.CreateBox(bounds)
    return obj


def create_line(p1, p2):
    """Create a 3D line from p1=[x1,y1,z1] to p2=[x2,y2,z2]"""
    obj = VTKObject()
    obj.CreateLine(p1, p2)
    return obj


def create_lines(lines, line_color=(1.0, 1.0, 1.0), line_width=1):
    """Create multiple 3D lines

    Args:
        lines: List of 3D lines, each element is [x1, y1, z1, x2, y2, z2]
    """
    lines = _convert_torch_to_numpy(lines)

    obj = VTKObject()
    obj.CreateLines(lines, line_color, line_width)
    return obj

