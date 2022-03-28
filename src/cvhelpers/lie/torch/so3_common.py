import torch

from .utils import allclose, isclose

_PI = 3.141592653589793


def is_valid_quaternion(q: torch.tensor) -> bool:
    return allclose(torch.norm(q, dim=-1), 1.0)


def normalize_quaternion(q: torch.tensor) -> torch.tensor:
    return q / torch.norm(q, dim=-1, keepdim=True)


def is_valid_rotmat(mat) -> bool:
    """Checks if matrix is a valid rotation"""

    with torch.no_grad():
        # Determinants of each matrix in the batch should be 1
        det_check = allclose(torch.det(mat), 1.)

        # The transpose of each matrix in the batch should be its inverse
        inv_check = allclose(mat.transpose(-1, -2) @ mat,
                             torch.eye(3, device=mat.device, dtype=mat.dtype)[None, :, :])

    return det_check and inv_check


def normalize_rotmat(mat: torch.Tensor) -> torch.Tensor:
    """Normalizes rotation matrix to a valid one"""
    u, _, v = torch.svd(mat)

    s = torch.zeros_like(mat)
    s[..., 0, 0] = 1.0
    s[..., 1, 1] = 1.0
    s[..., 2, 2] = torch.det(u) * torch.det(v)
    return u @ s @ v.transpose(-1, -2)


def hat(v: torch.Tensor):
    """Maps a vector to a 3x3 skew symmetric matrix."""
    h = torch.zeros((*v.shape, 3), dtype=v.dtype, device=v.device)
    h[..., 0, 1] = -v[..., 2]
    h[..., 0, 2] = v[..., 1]
    h[..., 1, 2] = -v[..., 0]
    h = h - h.transpose(-1, -2)
    return h


def vee(mat: torch.Tensor):
    """Inverse of hat operator, i.e. transforms skew-symmetric matrix to
    3-vector
    """
    v = torch.stack([
        mat[..., 2, 1],
        mat[..., 0, 2],
        mat[..., 1, 0],
    ], dim=-1)
    return v


def quat2rotmat(quat: torch.Tensor) -> torch.Tensor:
    """From a rotation matrix from a unit length quaternion
    Note that quaternion ordering is 'wxyz'.
    """
    assert quat.shape[-1] == 4
    qw, qx, qy, qz = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    qx2, qy2, qz2 = qx * qx, qy * qy, qz * qz

    # Form the matrix
    R00 = 1. - 2. * (qy2 + qz2)
    R01 = 2. * (qx * qy - qw * qz)
    R02 = 2. * (qw * qy + qx * qz)

    R10 = 2. * (qw * qz + qx * qy)
    R11 = 1. - 2. * (qx2 + qz2)
    R12 = 2. * (qy * qz - qw * qx)

    R20 = 2. * (qx * qz - qw * qy)
    R21 = 2. * (qw * qx + qy * qz)
    R22 = 1. - 2. * (qx2 + qy2)

    R0 = torch.stack([R00, R01, R02], dim=-1)
    R1 = torch.stack([R10, R11, R12], dim=-1)
    R2 = torch.stack([R20, R21, R22], dim=-1)
    mat = torch.stack([R0, R1, R2], dim=-2)
    return mat


def rotmat2quat(mat: torch.Tensor) -> torch.Tensor:
    """Converts rotation matrix to quaternion.
    This uses the algorithm found on
    https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    , and follows the code from ceres-solver
    https://github.com/ceres-solver/ceres-solver/blob/master/include/ceres/rotation.h
    """
    mat_shape = mat.shape
    assert mat_shape[-2:] == (3, 3)

    mat = torch.reshape(mat, [-1, 3, 3])

    # Case A: Easy case
    r = torch.sqrt(torch.clamp_min(1. + mat[:, 0, 0] + mat[:, 1, 1] + mat[:, 2, 2], 0.0))
    s = 0.5 / r
    quat = torch.stack([
        0.5 * r,
        (mat[:, 2, 1] - mat[:, 1, 2]) * s,
        (mat[:, 0, 2] - mat[:, 2, 0]) * s,
        (mat[:, 1, 0] - mat[:, 0, 1]) * s
    ], dim=-1)

    near_pi = isclose(r, 0.0)
    if torch.sum(near_pi) > 0:
        # Case B0, B1, B2: ~180deg rotation
        quats1 = mat.new_zeros([mat.shape[0], 3, 4])
        case_idx = torch.argmax(torch.diagonal(mat, dim1=-1, dim2=-2), dim=-1)
        for case, (i, j, k) in enumerate([[0, 1, 2], [1, 2, 0], [2, 0, 1]]):
            r = torch.sqrt(mat[..., i, i] - mat[..., j, j] - mat[..., k, k] + 1.0)
            s = 0.5 / r
            quats1[:, case, 0] = (mat[:, k, j] - mat[:, j, k]) * s
            quats1[:, case, i + 1] = 0.5 * r
            quats1[:, case, j + 1] = (mat[:, i, j] + mat[:, j, i]) * s
            quats1[:, case, k + 1] = (mat[:, k, i] + mat[:, i, k]) * s
        quat1 = quats1[torch.arange(mat.shape[0]), case_idx, :]
        quat[near_pi] = quat1[near_pi]

    quat = torch.reshape(quat, [*mat_shape[:-2], 4])
    return quat


def quat_inv(quat: torch.Tensor):
    """Quaternion inverse, which is equivalent to its conjugate"""
    assert quat.shape[-1] == 4

    inv = torch.cat([quat[..., 0:1], -quat[..., 1:]], dim=-1)
    return inv


def quat_mul(q1: torch.Tensor, q2: torch.Tensor):
    """Computes qout = q1 * q2, where * is the Hamilton product between the two
    quaternions. Note that the Hamiltonian product is not commutative.

    Args:
        q1: Quaternions of shape ([*, ], 4)
        q2: Quaternions of shape ([*, ], 4)

    Returns:
        qout = q1*q2.
    """
    assert q1.shape[-1] == 4 and q2.shape[-1] == 4

    qout = torch.stack([
        q1[..., 0] * q2[..., 0] - q1[..., 1] * q2[..., 1] - q1[..., 2] * q2[..., 2] - q1[..., 3] * q2[..., 3],
        q1[..., 0] * q2[..., 1] + q1[..., 1] * q2[..., 0] + q1[..., 2] * q2[..., 3] - q1[..., 3] * q2[..., 2],
        q1[..., 0] * q2[..., 2] - q1[..., 1] * q2[..., 3] + q1[..., 2] * q2[..., 0] + q1[..., 3] * q2[..., 1],
        q1[..., 0] * q2[..., 3] + q1[..., 1] * q2[..., 2] - q1[..., 2] * q2[..., 1] + q1[..., 3] * q2[..., 0]
    ], dim=-1)
    return qout


def quat_rot(quat: torch.Tensor, pt: torch.Tensor):
    # Pad point to form a (pure) quaternion
    zeros = torch.zeros_like(pt[..., 0:1])
    v = torch.cat([zeros, pt], dim=-1)

    quat2 = quat[..., None, :]
    rotated = quat_mul(quat_mul(quat2, v), quat_inv(quat2))
    return rotated[..., 1:]


def uniform_2_sphere(size: int = 1, device=None):
    """Uniform sampling on a 2-sphere
    Follows the algorithm from https://gist.github.com/andrewbolster/10274979

    Args:
        size: Number of vectors to sample

    Returns:
        Random Vector (np.ndarray) of size (size, 3) with norm 1.
            If size is None returned value will have size (3,)

    """
    phi = torch.rand(size, device=device) * 2 * _PI
    cos_theta = torch.rand(size, device=device) * 2.0 - 1.0
    theta = torch.arccos(cos_theta)
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)

    return torch.stack((x, y, z), dim=-1)



def sample_uniform_rot(size: int = 1, device=None):
    """We sample a random quaternion and convert it to 3x3 matrix"""
    u1 = torch.rand(size, device=device)
    u2 = torch.rand(size, device=device) * 2.0 * _PI
    u3 = torch.rand(size, device=device) * 2.0 * _PI

    a = torch.sqrt(1 - u1)
    b = torch.sqrt(u1)

    q = torch.stack([a * torch.sin(u2), a * torch.cos(u2),
                     b * torch.sin(u3), b * torch.cos(u3)], dim=-1)
    return quat2rotmat(q)