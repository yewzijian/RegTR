import numpy as np

from . import so3_common as so3c


def is_valid_quat_trans(vec: np.ndarray) -> bool:
    """7D vec contains a valid quaternion"""
    assert vec.shape[-1] == 7
    return so3c.is_valid_quaternion(vec[..., :4])


def normalize_quat_trans(vec: np.ndarray) -> np.ndarray:
    """Normalizes SE(3) &D vec to have a valid rotation component"""

    trans = vec[..., 4:]
    rot = so3c.normalize_quaternion(vec[..., :4])

    vec = np.concatenate([rot, trans], axis=-1)
    return vec


def is_valid_matrix(mat: np.ndarray) -> bool:
    """Checks if 4x4 matrix is a valid SE(3) matrix"""
    return so3c.is_valid_rotmat(mat[..., :3, :3])


def normalize_matrix(mat: np.ndarray) -> np.ndarray:
    """Normalizes SE(3) matrix to have a valid rotation component"""
    trans = mat[..., :3, 3:]
    rot = so3c.normalize_rotmat(mat[..., :3, :3])

    mat = np.concatenate([rot, trans], axis=-1)
    bottom_row = np.zeros_like(mat[..., :1, :])
    bottom_row[..., -1, -1] = 1.0
    return np.concatenate([mat, bottom_row], axis=-2)


def hat(v: np.ndarray) -> np.ndarray:
    """hat-operator for SE(3)
    Specifically, it takes in the 6-vector representation (= twist) and returns
    the corresponding matrix representation of Lie algebra element.

    Args:
        v: Twist vector of size ([*,] 6). As with common convention, first 3
           elements denote translation.

    Returns:
        mat: se(3) element of size ([*,] 4, 4)
    """
    mat = np.zeros((*v.shape[:-1], 4, 4))
    mat[..., :3, :3] = so3c.hat(v[..., 3:])  # Rotation
    mat[..., :3, 3] = v[..., :3]  # Translation

    return mat


def vee(mat: np.ndarray) -> np.ndarray:
    """vee-operator for SE(3), i.e. inverse of hat() operator.

    Args:
        mat: ([*, ] 4, 4) matrix containing the 4x4-matrix lie algebra
             representation. Omega must have the following structure:
                 |  0 -f  e  a |
                 |  f  0 -d  b |
                 | -e  d  0  c |
                 |  0  0  0  0 | .

    Returns:
        v: twist vector of size ([*,] 6)

    """
    v = np.zeros((*mat.shape[:-2], 6))
    v[..., 3:] = so3c.vee(mat[..., :3, :3])
    v[..., :3] = mat[..., :3, 3]
    return v


def quattrans2mat(vec: np.ndarray) -> np.ndarray:
    """Convert 7D quaternion+translation to a 4x4 SE(3) matrix"""
    rot, trans = vec[..., :4], vec[..., 4:]
    rotmat = so3c.quat2rotmat(rot)
    top = np.concatenate([rotmat, trans[..., None]], axis=-1)
    bottom_row = np.zeros_like(top[..., :1, :])
    bottom_row[..., -1, -1] = 1.0
    mat = np.concatenate([top, bottom_row], axis=-2)
    return mat


def mat2quattrans(mat: np.ndarray) -> np.ndarray:
    """Convert  4x4 SE(3) matrix to 7D quaternion+translation"""
    assert mat.shape[-2:] == (4, 4), 'Matrix should be of shape ([*,] 4, 4)'
    quat = so3c.rotmat2quat(mat[..., :3, :3]).data
    trans = mat[..., :3, 3]
    vec = np.concatenate([quat, trans], axis=-1)
    return vec
