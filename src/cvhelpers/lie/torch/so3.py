from math import pi
from typing import Dict

import torch

from . import so3_common as so3c
from .liegroupbase import LieGroupBase
from .utils import isclose


class SO3(LieGroupBase):

    DIM = 9
    DOF = 3
    N = 3  # Group transformation is 3x3 matrices
    name = 'SO3Torch'

    @staticmethod
    def identity(size: int = None, dtype=None, device=None) -> 'SO3':
        if size is None:
            return SO3(torch.eye(3, 3, dtype=dtype, device=device))
        else:
            return SO3(torch.eye(3, 3, dtype=dtype, device=device)[None, ...].repeat(size, 1, 1))

    @staticmethod
    def sample_uniform(size: int = 1, device=None) -> 'SO3':
        rand_R = so3c.sample_uniform_rot(size, device)
        return SO3(rand_R)

    @staticmethod
    def from_matrix(mat: torch.Tensor) -> 'SO3':
        assert mat.shape[-2:] == (3, 3), 'Matrix should be of shape ([*,] 3, 3)'
        assert so3c.is_valid_rotmat(mat), 'Matrix is not a valid rotation'
        return SO3(mat)

    @staticmethod
    def from_quaternion(quat: torch.Tensor, normalize: bool = False) -> 'SO3':
        """From a rotation matrix from a unit length quaternion
        Note that quaternion ordering is 'wxyz'.
        """
        if normalize:
            quat = quat / torch.norm(quat, dim=-1)
        return SO3(so3c.quat2rotmat(quat))

    def inv(self) -> 'SO3':
        irot = self.data[..., 0:3, 0:3].transpose(-1, -2)
        return SO3(irot)

    @staticmethod
    def exp(omega: torch.Tensor) -> 'SO3':
        """Group exponential. Converts an element of tangent space (i.e. rotation
        vector) representation to rotation matrix using Rodrigues rotation formula.

        To be specific, computes expm(hat(omega)) with expm being the matrix
        exponential and hat() is as defined above

        Args:
           omega: Rotation vector representation of shape ([N, ] 3)

        Returns:
           rotation matrix of size ([N, ] 3, 3)
        """
        rot, _ = SO3.exp_and_theta(omega)
        return rot

    @staticmethod
    def pexp(omega: torch.Tensor) -> 'SO3':
        return SO3.exp(omega)  # Pseudo log/exp only for SE(3).

    @staticmethod
    def exp_and_theta(omega: torch.Tensor) -> ('SO3', torch.Tensor):
        """Same as exp() but also returns theta (rotation angle in radians)
        """
        assert omega.shape[-1] == 3, 'Omega should have size (*, 3)'
        orig_shape = omega.shape
        if omega.ndim == 1:
            omega = omega[None, :]

        theta: torch.Tensor = torch.norm(omega, dim=-1, keepdim=True)
        rot_mat = omega.new_zeros((*omega.shape, 3))
        near_zero = isclose(theta, 0.)[:, 0]
        large_angle = near_zero.logical_not()

        # Near phi==0, use first order Taylor expansion
        if near_zero.sum() > 0:
            omega_0 = omega[near_zero, :]
            rot_mat[near_zero, :, :] = \
                torch.eye(3, dtype=omega.dtype, device=omega.device) + SO3.hat(omega_0)

        # Otherwise, use Rodrigues formulae
        if large_angle.sum() > 0:
            theta_large_angles = theta[large_angle, :]
            omega_1 = omega[large_angle, :]
            w = omega_1 / theta_large_angles  # axis, with norm = 1
            w_hat = SO3.hat(w)
            w_hat2 = w_hat @ w_hat
            s = torch.sin(theta_large_angles)[..., None]
            c = torch.cos(theta_large_angles)[..., None]
            rot_mat[large_angle, :, :] = \
                torch.eye(3, dtype=omega.dtype, device=omega.device) \
                + s * w_hat + (1 - c) * w_hat2

        rot_mat = torch.reshape(rot_mat, (*orig_shape, 3))
        return SO3(rot_mat), theta

    def log(self) -> torch.Tensor:
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
        orig_shape = mat.shape
        if mat.ndim == 2:
            mat = mat[None, :]

        # Computes k * 2 * sin(theta) where k is axis of rotation
        angle_axis = torch.stack([mat[..., 2, 1] - mat[..., 1, 2],
                                  mat[..., 0, 2] - mat[..., 2, 0],
                                  mat[..., 1, 0] - mat[..., 0, 1]], dim=-1)
        diag = torch.stack([mat[..., 0, 0],
                            mat[..., 1, 1],
                            mat[..., 2, 2]], dim=-1)
        trace = torch.sum(diag, dim=-1, keepdim=True)
        cos_theta = torch.clamp(0.5 * (trace - 1), min=-1.0, max=1.0)
        sin_theta = torch.clamp(0.5 * torch.norm(angle_axis, dim=-1, keepdim=True), max=1.0)
        theta = torch.atan2(sin_theta, cos_theta)

        near_pi = isclose(theta, pi)[:, 0]
        near_zero = isclose(theta, 0.)[:, 0]
        not_zero_pi = torch.logical_or(near_pi, near_zero).logical_not()
        vec = mat.new_zeros(mat.shape[:-1])

        # Case 1: angle ~ 0: sin(theta) ~ theta is good approximation (taylor)
        if near_zero.sum() > 0:
            vec[near_zero] = 0.5 * angle_axis[near_zero]
        # Case 2: Usual formula
        if not_zero_pi.sum() > 0:
            r = 0.5 * theta[not_zero_pi] / sin_theta[not_zero_pi]
            vec[not_zero_pi] = r * angle_axis[not_zero_pi]
        # Case 3: angle ~ pi. This is the hard case. Since theta is large,
        # and sin(theta) is small. Dividing by theta by sin(theta) will
        # either give an overflow, or worse still, numerically meaningless
        # results.
        if near_pi.sum() > 0:
            vec_pi = theta[near_pi] * torch.sqrt((diag[near_pi] - cos_theta[near_pi]) / (1.0 - cos_theta[near_pi]))
            sign = torch.sign(angle_axis[near_pi] * sin_theta[near_pi])
            sign[sign == 0] = 1
            vec_pi = vec_pi * sign
            vec[near_pi] = vec_pi

        vec = vec.reshape(orig_shape[:-1])
        return vec

    def transform(self, pts: torch.Tensor) -> torch.Tensor:
        assert len(self.shape) == pts.ndim - 2
        ptsT = pts.transpose(-1, -2)
        transformedT = self.data @ ptsT
        transformed = transformedT.transpose(-1, -2)
        return transformed

    @staticmethod
    def hat(v: torch.Tensor):
        """Maps a vector to a 3x3 skew symmetric matrix."""
        return so3c.hat(v)

    @staticmethod
    def vee(mat: torch.Tensor):
        """Inverse of hat operator, i.e. transforms skew-symmetric matrix to
        3-vector
        """
        return so3c.vee(mat)

    """Comparison functions"""
    def rotation_angle(self) -> torch.Tensor:
        """Returns the rotation angle in radians"""
        mat = self.data
        trace = mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2]
        rot_err_rad = torch.acos(torch.clamp(0.5 * (trace - 1), min=-1.0, max=1.0))
        return rot_err_rad

    def compare(self, other: 'SO3') -> Dict:
        """Compares two SO3 instances, returning the rotation error in degrees"""
        error = self * other.inv()
        e = {'rot_deg': SO3.rotation_angle(error) * 180 / pi}
        return e

    """Conversion functions"""
    def vec(self) -> torch.Tensor:
        """Returns the flattened representation"""
        return self.data.transpose(-1, -2).reshape(*self.data.shape[:-2], 9)

    def as_quaternion(self) -> torch.Tensor:
        return so3c.rotmat2quat(self.data)

    def as_matrix(self) -> torch.Tensor:
        return self.data

    def is_valid(self) -> bool:
        """Check whether the data is valid, e.g. if the underlying SE(3)
        representation has a valid rotation"""
        return so3c.is_valid_rotmat(self.data)

    def make_valid(self):
        """Rectifies the data so that the representation is valid"""
        return SO3(so3c.normalize_rotmat(self.data))
