from math import pi
from typing import Dict, Union

import torch

from . import se3_common as se3c
from .so3 import SO3
from .liegroupbase import LieGroupBase
from .utils import isclose


class SE3(LieGroupBase):

    DIM = 12
    DOF = 6
    N = 4  # Group transformation is 4x4 matrices
    name = 'SE3Torch'

    @staticmethod
    def identity(size: int = None, dtype=None, device=None) -> 'SE3':
        if size is None:
            return SE3(torch.eye(4, 4, dtype=dtype, device=device))
        else:
            return SE3(torch.eye(4, 4, dtype=dtype, device=device)[None, ...].repeat(size, 1, 1))

    @staticmethod
    def _from_rt(rot: Union[SO3, torch.Tensor], trans: torch.Tensor) -> 'SE3':
        """Convenience function to concatenates the rotation and translation
        part into a SE(3) matrix

        Args:
            rot: ([*,] 3, 3) or SO3
            trans: ([*,] 3, 1)

        Returns:
            SE(3) matrix
        """
        rot_mat: torch.Tensor = rot if isinstance(rot, torch.Tensor) else rot.data
        mat = torch.cat([rot_mat, trans], dim=-1)
        bottom_row = torch.zeros_like(mat[..., :1, :])
        bottom_row[..., -1, -1] = 1.0
        mat = torch.cat([mat, bottom_row], dim=-2)
        return SE3(mat)

    @staticmethod
    def from_rtvec(vec: torch.Tensor) -> 'SE3':
        """Constructs from 7D vector"""
        assert se3c.is_valid_quat_trans(vec)
        return SE3(se3c.quattrans2mat(vec))

    @staticmethod
    def from_matrix(mat: torch.Tensor) -> 'SE3':
        assert mat.shape[-2:] == (4, 4), 'Matrix should be of shape ([*,] 4, 4)'
        assert se3c.is_valid_matrix(mat), 'Matrix is not a valid rotation'
        return SE3(mat)

    def inv(self) -> 'SE3':
        rot = self.data[..., :3, :3]  # (3, 3)
        trans = self.data[..., :3, 3:]  # (3, 1)

        inv_rot = rot[..., 0:3, 0:3].transpose(-1, -2)
        return SE3._from_rt(inv_rot, inv_rot @ -trans)

    @staticmethod
    def exp(vec: torch.Tensor) -> 'SE3':
        """Group exponential. Converts an element of tangent space (twist) to the
        corresponding element of the group SE(3).

        To be specific, computes expm(hat(psi)) with expm being the matrix
        exponential and hat() being the hat operator of SE(3).

        Args:
            vec: Twist vector ([N, ] 6)

        Returns:
            SE(3) matrix of size ([N, ] 4, 4)

        Credits: Implementation is inspired by that in Sophus library
                 https://github.com/strasdat/Sophus/blob/master/sophus/se3.hpp
        """
        orig_shape = vec.shape
        if vec.ndim == 1:
            vec = vec[None, :]

        v, omega = vec[..., :3], vec[..., 3:]
        rot_mat, theta = SO3.exp_and_theta(omega)
        theta = theta[..., None]
        near_zero = isclose(theta, 0.)[:, 0, 0]
        large_angle = near_zero.logical_not()
        s, c = torch.sin(theta), torch.cos(theta)

        Omega = SO3.hat(omega)
        Omega_sq = Omega @ Omega
        theta2, theta3 = theta ** 2, theta ** 3

        V = vec.new_zeros((*vec.shape[:-1], 3, 3))

        # Case 1: General case
        if large_angle.sum() > 0:
            V[large_angle, :, :] = torch.eye(3, device=vec.device, dtype=vec.dtype) \
                                   - (c[large_angle] - 1.0) / theta2[large_angle] * Omega[large_angle] \
                                   + (theta[large_angle] - s[large_angle]) / theta3[large_angle] * Omega_sq[large_angle]
        # Case 2: theta ~ 0
        if near_zero.sum() > 0:
            V[near_zero] = rot_mat.data[near_zero]

        trans = V @ v[..., None]
        retval = SE3._from_rt(rot_mat, trans)
        retval.data = torch.reshape(retval.data, (*orig_shape[:-1], 4, 4))

        return retval

    @staticmethod
    def pexp(vec: torch.Tensor) -> 'SE3':
        """Group pseudo-exponential. Converts an element of pseudo-log space
        (twist) to the corresponding element of the group SE(3).

        Similar to exp(), but leaves the translation portion intact

        Args:
            vec: Twist vector ([N, ] 6)

        Returns:
            SE(3) matrix of size ([N, ] 4, 4)
        """
        orig_shape = vec.shape
        if vec.ndim == 1:
            vec = vec[None, :]

        v, omega = vec[..., :3], vec[..., 3:]
        rot_mat, theta = SO3.exp_and_theta(omega)
        retval = SE3._from_rt(rot_mat, v[..., None])
        retval.data = torch.reshape(retval.data, (*orig_shape[:-1], 4, 4))

        return retval

    def log(self) -> torch.Tensor:
        """Logarithm map.
        """
        raise NotImplementedError

    def transform(self, pts: torch.tensor) -> torch.tensor:
        assert len(self.shape) == pts.ndim - 2
        ptsT = pts.transpose(-1, -2)
        transformedT = self.data[..., :3, :3] @ ptsT + self.data[..., :3, 3:4]
        transformed = transformedT.transpose(-1, -2)
        return transformed

    @staticmethod
    def hat(v: torch.Tensor):
        """hat-operator for SE(3)
        Specifically, it takes in the 6-vector representation (= twist) and returns
        the corresponding matrix representation of Lie algebra element.

        Args:
            v: Twist vector of size ([*,] 6). As with common convention, first 3
               elements denote translation.

        Returns:
            mat: se(3) element of size ([*,] 4, 4)
        """
        return se3c.hat(v)

    @staticmethod
    def vee(mat: torch.Tensor):
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
        return se3c.vee(mat)

    """Jacobians"""
    def jacob_expeD_de(poseD: 'SE3'):
        """Jacobian d (exp(eps) * D) / d eps , with eps=increment in Lie Algebra.

        See section 10.3.3. in [1]

        Args:
            poseD: SE(3) pose of size (B, 4, 4)

        Returns:
            Jacobian d (exp(eps) * D) / d eps
        """
        matD = poseD.data
        dc1_hat = SO3.hat(matD[..., :3, 0])
        dc2_hat = SO3.hat(matD[..., :3, 1])
        dc3_hat = SO3.hat(matD[..., :3, 2])
        dt_hat = SO3.hat(matD[..., :3, 3])

        jac = matD.new_zeros((*matD.shape[:-2], 12, 6))
        jac[..., 9, 0] = 1.0
        jac[..., 10, 1] = 1.0
        jac[..., 11, 2] = 1.0
        jac[..., 0:3, 3:6] = -dc1_hat
        jac[..., 3:6, 3:6] = -dc2_hat
        jac[..., 6:9, 3:6] = -dc3_hat
        jac[..., 9:12, 3:6] = -dt_hat

        return jac

    def jacob_Dexpe_de(poseD: 'SE3'):
        """Jacobian d (D * exp(eps)) / d eps , with eps=increment in Lie Algebra.

        See section 10.3.4. in [1]

        Args:
            poseD: SE(3) pose of size (B, 4, 4)

        Returns:
            Jacobian d (D * exp(eps)) / d eps
        """
        matD = poseD.data

        jac = matD.new_zeros((*matD.shape[:-2], 12, 6))
        jac[..., 9:12, 0:3] = matD[..., :3, :3]
        jac[..., 0:3, 4] = -matD[..., :3, 2]
        jac[..., 0:3, 5] = matD[..., :3, 1]
        jac[..., 3:6, 3] = matD[..., :3, 2]
        jac[..., 3:6, 5] = -matD[..., :3, 0]
        jac[..., 6:9, 3] = -matD[..., :3, 1]
        jac[..., 6:9, 4] = matD[..., :3, 0]
        return jac

    def jacob_dAexpeD_de(poseA: 'SE3', poseD: 'SE3', full_matrix: bool = True):
        """Jacobian d (A * exp(eps) * D) / d eps , with eps=increment in Lie Algebra.

        See section 10.3.7. in [1]

        Args:
            poseA: SE(3) pose of size (B, 4, 4)
            poseD: SE(3) pose of size (B, 4, 4)
            full_matrix: Whether to return the full jacobians with the zero elements
              If full_matrix=True, the output jacobian will have
              shape (B, 12, 6).
              Otherwise it'll have shape (B, 15, 3) containing the five 3x3 non-zero
              blocks of the jacobian. Specifically, output[i, :, :] is a 15x3s matrix
              of the form A,B,C,D,E where each is a 3x3 block and the full jacobian
              is given by |0 0 0 A|.transpose()
                          |B C D E|

        Returns:
            Jacobian d (A * exp(eps) * D) / d eps
        """
        matA, matD = poseA.data, poseD.data
        rotA = matA[..., :3, :3]
        dc1_hat = SO3.hat(matD[..., :3, 0])
        dc2_hat = SO3.hat(matD[..., :3, 1])
        dc3_hat = SO3.hat(matD[..., :3, 2])
        dt_hat = SO3.hat(matD[..., :3, 3])

        blockA = rotA
        blockB = -rotA @ dc1_hat
        blockC = -rotA @ dc2_hat
        blockD = -rotA @ dc3_hat
        blockE = -rotA @ dt_hat

        if full_matrix:
            jac = matA.new_zeros((*matA.shape[:-2], 12, 6))
            jac[..., 9:12, 0:3] = blockA
            jac[..., 0:3, 3:6] = blockB
            jac[..., 3:6, 3:6] = blockC
            jac[..., 6:9, 3:6] = blockD
            jac[..., 9:12, 3:6] = blockE
        else:
            jac = torch.cat([blockA, blockB, blockC, blockD, blockE], dim=-2)

        return jac

    """Comparison functions"""
    def compare(self, other: 'SE3') -> Dict:
        """Compares two SO3 instances, returning the rotation error in degrees
        Note that for the translation error, we compare the translation portion
        directly directly and not on the error term, to be consistent with
        "Learning Transformation Synchronization" (CVPR2019)
        """
        error = self * other.inv()
        e = {'rot_deg': SO3.rotation_angle(error.rot) * 180 / pi,
             'trans': torch.norm(self.trans - other.trans, dim=-1)}
        return e

    """Conversion functions"""
    @property
    def rot(self) -> SO3:
        return SO3(self.data[..., :3, :3])

    @property
    def trans(self) -> torch.Tensor:
        return self.data[..., :3, 3]

    def vec(self) -> torch.Tensor:
        """Returns the flattened representation, which follows the ordering
        in [1], section 7.1"""
        return self.data[..., :3, :].transpose(-1, -2).reshape(*self.data.shape[:-2], 12)

    def as_quat_trans(self) -> torch.Tensor:
        """Returns a 7D representation of the transformation, containing the
        4D quaternion (WXYZ) and the 3D translation"""
        return se3c.mat2quattrans(self.data)

    def as_matrix(self) -> torch.Tensor:
        return self.data

    def is_valid(self) -> bool:
        """Check whether the data is valid, e.g. if the underlying SE(3)
        representation has a valid rotation"""
        return se3c.is_valid_matrix(self.data)

    def make_valid(self) -> 'SE3':
        """Rectifies the data so that the representation is valid"""
        return SE3(se3c.normalize_matrix(self.data))
