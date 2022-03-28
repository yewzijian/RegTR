import numpy as np

from .liegroupbase import _EPS


def is_valid_quaternion(q: np.ndarray) -> bool:
    return np.allclose(np.linalg.norm(q, axis=-1), 1.0)


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def is_valid_rotmat(mat: np.ndarray) -> bool:
    """Checks if matrix is a valid rotation"""

    # We do not check elements which are nan, since those are known to be
    # invalid rotations
    not_nan = np.sum(np.isnan(mat), axis=(-1, -2)) == 0
    mat = mat[not_nan]

    # Determinants of each matrix in the batch should be 1
    det_check = np.allclose(np.linalg.det(mat), 1.)

    # The transpose of each matrix in the batch should be its inverse
    # We set a greater tolerane to handle outputs from float32 algorithms
    inv_check = np.allclose(mat.swapaxes(-1, -2) @ mat, np.eye(3)[None, :, :], atol=5e-5)

    return np.logical_and(det_check, inv_check)


def normalize_rotmat(mat: np.ndarray) -> np.ndarray:
    """Normalizes rotation matrix to a valid one"""
    u, _, vt = np.linalg.svd(mat)

    s = np.zeros_like(mat)
    s[..., 0, 0] = 1.0
    s[..., 1, 1] = 1.0
    s[..., 2, 2] = np.linalg.det(u) * np.linalg.det(vt)
    return u @ s @ vt


def hat(v: np.ndarray) -> np.ndarray:
    """Maps a vector to a 3x3 skew symmetric matrix."""
    h = np.zeros((*v.shape, 3))
    h[..., 0, 1] = -v[..., 2]
    h[..., 0, 2] = v[..., 1]
    h[..., 1, 2] = -v[..., 0]
    h = h - h.swapaxes(-1, -2)
    return h


def vee(mat: np.ndarray) -> np.ndarray:
    """Inverse of hat operator, i.e. transforms skew-symmetric matrix to
    3-vector
    """
    v = np.stack([
        mat[..., 2, 1],
        mat[..., 0, 2],
        mat[..., 1, 0],
    ], axis=-1)

    return v


def quat2rotmat(quat: np.ndarray, normalize: bool = False) -> np.ndarray:
    """From a rotation matrix from a unit length quaternion
    Note that quaternion ordering is 'wxyz'.
    """
    assert quat.shape[-1] == 4
    if normalize:
        quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    elif not np.allclose(np.linalg.norm(quat, axis=-1), 1.):
        raise AssertionError('Quaternion must be unit length')

    qw, qx, qy, qz = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    qx2, qy2, qz2 = qx * qx, qy * qy, qz * qz

    # Form the matrix
    r00 = 1. - 2. * (qy2 + qz2)
    r01 = 2. * (qx * qy - qw * qz)
    r02 = 2. * (qw * qy + qx * qz)

    r10 = 2. * (qw * qz + qx * qy)
    r11 = 1. - 2. * (qx2 + qz2)
    r12 = 2. * (qy * qz - qw * qx)

    r20 = 2. * (qx * qz - qw * qy)
    r21 = 2. * (qw * qx + qy * qz)
    r22 = 1. - 2. * (qx2 + qy2)

    r0 = np.stack([r00, r01, r02], axis=-1)
    r1 = np.stack([r10, r11, r12], axis=-1)
    r2 = np.stack([r20, r21, r22], axis=-1)
    mat = np.stack([r0, r1, r2], axis=-2)
    return mat


def rotmat2quat(mat: np.ndarray) -> np.ndarray:
    """Converts rotation matrix to quaternion.
    This uses the algorithm found on
    https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    , and follows the code from ceres-solver
    https://github.com/ceres-solver/ceres-solver/blob/master/include/ceres/rotation.h
    """
    mat_shape = mat.shape
    assert mat_shape[-2:] == (3, 3)
    assert is_valid_rotmat(mat), 'Matrix is not a valid SE(3)'

    mat = np.reshape(mat, [-1, 3, 3])

    # Case A: Easy case
    r = np.sqrt(np.maximum(1. + mat[:, 0, 0] + mat[:, 1, 1] + mat[:, 2, 2], 0.0))
    with np.errstate(divide='ignore', invalid='ignore'):  # Hide div0 warning
        s = 0.5 / r
        quat = np.stack([
            0.5 * r,
            (mat[:, 2, 1] - mat[:, 1, 2]) * s,
            (mat[:, 0, 2] - mat[:, 2, 0]) * s,
            (mat[:, 1, 0] - mat[:, 0, 1]) * s
        ], axis=-1)

    near_pi = r < _EPS

    if np.sum(near_pi) > 0:
        # Case B0, B1, B2: ~180deg rotation
        quats1 = np.zeros([mat.shape[0], 3, 4])
        case_idx = np.argmax(np.diagonal(mat, axis1=-1, axis2=-2), axis=-1)
        for case, (i, j, k) in enumerate([[0, 1, 2], [1, 2, 0], [2, 0, 1]]):
            r = np.sqrt(np.maximum(mat[..., i, i] - mat[..., j, j] - mat[..., k, k] + 1.0, 0.0))
            with np.errstate(divide='ignore', invalid='ignore'):  # Hide div0 warning
                s = 0.5 / r
                quats1[:, case, 0] = (mat[:, k, j] - mat[:, j, k]) * s
                quats1[:, case, i + 1] = 0.5 * r
                quats1[:, case, j + 1] = (mat[:, i, j] + mat[:, j, i]) * s
                quats1[:, case, k + 1] = (mat[:, k, i] + mat[:, i, k]) * s
        quat1 = quats1[np.arange(mat.shape[0]), case_idx, :]
        quat = np.where(near_pi[:, None], quat1, quat)

    quat = np.reshape(quat, [*mat_shape[:-2], 4])
    return quat


def quat_inv(quat: np.ndarray):
    """Quaternion inverse, which is equivalent to its conjugate"""
    assert quat.shape[-1] == 4

    inv = np.concatenate([quat[..., 0:1], -quat[..., 1:]], axis=-1)
    return inv


def quat_mul(q1: np.ndarray, q2: np.ndarray):
    """Computes qout = q1 * q2, where * is the Hamilton product between the two
    quaternions. Note that the Hamiltonian product is not commutative.

    Args:
        q1: Quaternions of shape ([*, ], 4)
        q2: Quaternions of shape ([*, ], 4)

    Returns:
        qout = q1*q2.
    """
    assert q1.shape[-1] == 4 and q2.shape[-1] == 4

    qout = np.stack([
        q1[..., 0] * q2[..., 0] - q1[..., 1] * q2[..., 1] - q1[..., 2] * q2[..., 2] - q1[..., 3] * q2[..., 3],
        q1[..., 0] * q2[..., 1] + q1[..., 1] * q2[..., 0] + q1[..., 2] * q2[..., 3] - q1[..., 3] * q2[..., 2],
        q1[..., 0] * q2[..., 2] - q1[..., 1] * q2[..., 3] + q1[..., 2] * q2[..., 0] + q1[..., 3] * q2[..., 1],
        q1[..., 0] * q2[..., 3] + q1[..., 1] * q2[..., 2] - q1[..., 2] * q2[..., 1] + q1[..., 3] * q2[..., 0]
    ], axis=-1)
    return qout


def quat_rot(quat: np.ndarray, pts: np.ndarray):
    """Rotate points"""

    # Pad point to form a (pure) quaternion
    zeros = np.zeros(pts[..., 0:1].shape)
    v = np.concatenate([zeros, pts], axis=-1)

    rotated = quat_mul(quat_mul(quat[..., None, :], v), quat_inv(quat[..., None, :]))
    return rotated[..., 1:]


def uniform_2_sphere(size: int = 1):
    """Uniform sampling on a 2-sphere

    Source: https://gist.github.com/andrewbolster/10274979

    Args:
        size: Number of vectors to sample

    Returns:
        Random Vector (np.ndarray) of size (size, 3) with norm 1.
        If size is None returned value will have size (3,)

    """
    if size is not None:
        phi = np.random.uniform(0.0, 2 * np.pi, size)
        cos_theta = np.random.uniform(-1.0, 1.0, size)
    else:
        phi = np.random.uniform(0.0, 2 * np.pi)
        cos_theta = np.random.uniform(-1.0, 1.0)

    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack((x, y, z), axis=-1)
