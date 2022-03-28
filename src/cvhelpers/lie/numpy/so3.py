from typing import Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from . import so3_common as so3c
from .liegroupbase import _EPS, LieGroupBase


class SO3(LieGroupBase):

    DIM = 9
    DOF = 3
    N = 3  # Group transformation is 3x3 matrices
    name = 'SO3Numpy'

    @staticmethod
    def identity(size: int = None) -> 'SO3':
        if size is None:
            return SO3(np.identity(3))
        else:
            return SO3(np.tile(np.identity(3)[None, ...], (size, 1, 1)))

    @staticmethod
    def sample_uniform(size: int = 1) -> 'SO3':
        # Use scipy's uniform rotation generator
        x = Rotation.random(size).as_matrix()
        return SO3(x)

    @staticmethod
    def sample_small(size: int = None, std=None) -> 'SO3':
        # First sample axis
        rand_dir = so3c.uniform_2_sphere(size)

        # Then samples angle magnitude
        theta = np.random.randn(size) if size is not None else np.random.randn()
        theta *= std * np.pi / np.sqrt(3)
        return SO3.exp(rand_dir * theta)

    @staticmethod
    def from_matrix(mat: np.ndarray, normalize: bool = False, check: bool = True) -> 'SO3':
        assert mat.shape[-2:] == (3, 3), 'Matrix should be of shape ([*,] 3, 3)'

        if normalize:
            normalized = so3c.normalize_rotmat(mat)
            assert np.allclose(normalized, mat, atol=1e-3), 'Provided matrix too far from being valid'
            return SO3(normalized)
        else:
            if check:
                assert so3c.is_valid_rotmat(mat), 'Matrix is not a valid rotation'
            return SO3(mat)

    @staticmethod
    def from_quaternion(quat, normalize: bool = False):
        """From a rotation matrix from a unit length quaternion
        Note that quaternion ordering is 'wxyz'.
        """
        return SO3(so3c.quat2rotmat(quat, normalize=normalize))

    def inv(self) -> 'SO3':
        irot = self.data[..., 0:3, 0:3].swapaxes(-1, -2)
        return SO3(irot)

    @staticmethod
    def exp(omega: np.ndarray) -> 'SO3':
        """Group exponential. Converts an element of tangent space (i.e. rotation
        vector) representation to rotation matrix using Rodrigues rotation formula.

        To be specific, computes expm(hat(omega)) with expm being the matrix
        exponential and hat() is as defined above

        Args:
            omega: Rotation vector representation of shape ([N, ] 3)

        Returns:
            rotation matrix of size ([N, ] 3, 3)
        """
        rotmat, _ = SO3.exp_and_theta(omega)
        return rotmat

    @staticmethod
    def exp_and_theta(omega: np.ndarray) -> ('SO3', np.ndarray):
        """Same as exp() but also returns theta (rotation angle in radians)
        """
        theta = np.linalg.norm(omega, axis=-1, keepdims=True)
        near_zero = np.isclose(theta, 0.)[..., None]

        # Near phi==0, use first order Taylor expansion
        rotmat_taylor = np.identity(3) + SO3.hat(omega)

        # Otherwise, use Rodrigues formulae
        with np.errstate(divide='ignore', invalid='ignore'):
            w = omega / theta  # axis, with norm = 1
        w_hat = SO3.hat(w)
        w_hat2 = w_hat @ w_hat
        s = np.sin(theta)[..., None]
        c = np.cos(theta)[..., None]
        rotmat_rodrigues = np.identity(3) + s * w_hat + (1 - c) * w_hat2

        rotmat = np.where(near_zero, rotmat_taylor, rotmat_rodrigues)
        return SO3(rotmat), theta

    def log(self) -> np.ndarray:
        """Logarithm map. Computes the logarithm, the inverse of the group
         exponential, mapping elements of the group (rotation matrices) to
         the tangent space (rotation-vector) representation.

        The conversion is numerically problematic when the rotation angle is close
        to zero or pi. We use the 3 branch implementation, similar to ceres solver,
        since most existing implementations do not handle the ~180 case.

        https://github.com/kashif/ceres-solver/blob/master/include/ceres/rotation.h

        Returns:
            rotation matrix of size ([N, ] 3)
        """

        mat = self.data

        # Computes k * 2 * sin(theta) where k is axis of rotation
        angle_axis = np.stack([mat[..., 2, 1] - mat[..., 1, 2],
                               mat[..., 0, 2] - mat[..., 2, 0],
                               mat[..., 1, 0] - mat[..., 0, 1]], axis=-1)

        diag = np.stack([mat[..., 0, 0],
                         mat[..., 1, 1],
                         mat[..., 2, 2]], axis=-1)
        trace = np.sum(diag, axis=-1, keepdims=True)
        cos_theta = np.clip(0.5 * (trace - 1), a_min=-1.0, a_max=1.0)
        sin_theta = np.minimum(0.5 * np.linalg.norm(angle_axis, axis=-1, keepdims=True),
                               1.0)
        theta = np.arctan2(sin_theta, cos_theta)

        near_zero_or_pi = np.abs(sin_theta) < _EPS
        near_zero = np.abs(theta) < _EPS

        # Case 1: angle ~ 0: sin(theta) ~ theta is good approximation (taylor)
        vec_taylor = 0.5 * angle_axis
        # Case 2: Usual formula (divide-by-zero warning suppressed)
        with np.errstate(divide='ignore', invalid='ignore'):
            r = 0.5 * theta / sin_theta
            vec_usual = r * angle_axis
        # Case 3: angle ~ pi. This is the hard case. Since theta is large,
        # and sin(theta) is small. Dividing by theta by sin(theta) will
        # either give an overflow, or worse still, numerically meaningless
        # results.
        with np.errstate(divide='ignore', invalid='ignore'):
            vec_pi = theta * np.sqrt((diag - cos_theta) / (1.0 - cos_theta))
            vec_pi[angle_axis * sin_theta < 0] *= -1

        vec = np.where(near_zero_or_pi, vec_pi, vec_usual)
        vec = np.where(near_zero, vec_taylor, vec)

        return vec

    def transform(self, pts: np.ndarray) -> np.ndarray:
        assert len(self.shape) == pts.ndim - 2
        ptsT = pts.swapaxes(-1, -2)
        transformedT = self.data @ ptsT
        transformed = transformedT.swapaxes(-1, -2)
        return transformed

    @staticmethod
    def hat(v: np.ndarray) -> np.ndarray:
        """Maps a vector to a 3x3 skew symmetric matrix."""
        return so3c.hat(v)

    @staticmethod
    def vee(mat: np.ndarray) -> np.ndarray:
        """Inverse of hat operator, i.e. transforms skew-symmetric matrix to
        3-vector
        """
        return so3c.vee(mat)

    """Comparison functions"""
    def rotation_angle(self) -> np.ndarray:
        """Returns the rotation angle in radians"""
        trace = np.trace(self.data, axis1=-1, axis2=-2)
        rot_err_rad = np.arccos(np.clip(0.5 * (trace - 1), a_min=-1.0, a_max=1.0))
        return rot_err_rad

    def compare(self, other: 'SO3') -> Dict:
        """Compares two SO3 instances, returning the rotation error in degrees"""
        error = self * other.inv()
        e = {'rot_deg': SO3.rotation_angle(error) * 180 / np.pi}
        return e

    """Conversion functions"""
    def vec(self) -> np.ndarray:
        """Returns the flattened representation"""
        return self.data.swapaxes(-1, -2).reshape(*self.data.shape[:-2], 9)

    def as_quaternion(self) -> np.ndarray:
        return so3c.rotmat2quat(self.data)

    def as_matrix(self) -> np.ndarray:
        return self.data

    def is_valid(self) -> bool:
        """Check whether the data is valid, e.g. if the underlying SE(3)
        representation has a valid rotation"""
        return so3c.is_valid_rotmat(self.data)
