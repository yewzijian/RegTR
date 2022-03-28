from typing import Dict, Union

import numpy as np

from . import se3_common as se3c
from .liegroupbase import _EPS, LieGroupBase
from .so3 import SO3


class SE3(LieGroupBase):

    DIM = 12
    DOF = 6
    N = 4  # Group transformation is 4x4 matrices
    name = 'SE3Numpy'

    @staticmethod
    def identity(size: int = None) -> 'SE3':
        if size is None:
            return SE3(np.identity(4))
        else:
            return SE3(np.tile(np.identity(4)[None, ...], (size, 1, 1)))

    @staticmethod
    def sample_uniform(size: int = 1) -> 'SE3':
        """Random poses. Rotation portion is uniformly generated, translation
        part is sampled from a unit cube with sides [-1, 1]
        """
        # Use scipy's uniform rotation generator
        rot = SO3.sample_uniform(size)
        if size is None:
            trans = (np.random.rand(3, 1) - 0.5) * 2
        else:
            trans = (np.random.randn(size, 3, 1) - 0.5) * 2
        return SE3._from_rt(rot, trans)

    @staticmethod
    def sample_small(size: int = None, std=None) -> 'SE3':

        rot = SO3.sample_small(size, std)
        trans = np.random.randn(3, 1) * std / np.sqrt(3) if size is None else \
            np.random.randn(size, 3, 1) * std / np.sqrt(3)

        return SE3._from_rt(rot, trans)

    @staticmethod
    def _from_rt(rot: Union[SO3, np.ndarray], trans: np.ndarray) -> 'SE3':
        """Convenience function to concatenates the rotation and translation
        part into a SE(3) matrix

        Args:
            rot: ([*,] 3, 3) or SO3
            trans: ([*,] 3, 1)

        Returns:
            SE(3) matrix
        """
        rot_mat: np.ndarray = rot if isinstance(rot, np.ndarray) else rot.data
        mat = np.concatenate([rot_mat, trans], axis=-1)
        bottom_row = np.zeros_like(mat[..., :1, :])
        bottom_row[..., -1, -1] = 1.0
        mat = np.concatenate([mat, bottom_row], axis=-2)
        return SE3(mat)
    
    @staticmethod
    def from_rtvec(vec: np.ndarray, normalize: bool = False) -> 'SE3':
        """Constructs from 7D vector"""
        if normalize:
            normalized = se3c.normalize_quat_trans(vec)
            assert np.allclose(normalized, vec, atol=1e-3), 'Provided vec is too far from valid'
            return SE3(se3c.quattrans2mat(normalized))
        else:
            assert se3c.is_valid_quat_trans(vec)
            return SE3(se3c.quattrans2mat(vec))

    @staticmethod
    def from_matrix(mat, normalize=False, check=True) -> 'SE3':
        assert mat.shape[-2:] in [(4, 4), (3, 4)], 'Matrix should be of shape ([*,] 3/4, 4)'
        if normalize:
            normalized = se3c.normalize_matrix(mat)
            # Ensure that the matrix isn't nonsense in the first place
            assert np.allclose(normalized, mat, atol=1e-3), 'Original SE3 is too far from being valid'
            return SE3(normalized)
        else:
            if check:
                assert se3c.is_valid_matrix(mat), 'Matrix is not a valid SE(3)'
            if mat.shape[-2:] == (3, 4):
                bottom_row = np.zeros_like(mat[..., :1, :])
                bottom_row[..., -1, -1] = 1.0
                mat = np.concatenate([mat, bottom_row], axis=-2)
            return SE3(mat)

    def inv(self) -> 'SE3':
        rot = self.data[..., :3, :3]  # (3, 3)
        trans = self.data[..., :3, 3:]  # (3, 1)
        inv_rot = np.swapaxes(rot, -1, -2)
        return SE3._from_rt(inv_rot, inv_rot @ -trans)

    @staticmethod
    def exp(vec: np.ndarray) -> 'SE3':
        """Group exponential. Converts an element of tangent space (twist) to the
        corresponding element of the group SE(3).

        To be specific, computes expm(hat(vec)) with expm being the matrix
        exponential and hat() being the hat operator of SE(3).

        Args:
            vec: Twist vector ([N, ] 6)

        Returns:
            SE(3) matrix of size ([N, ] 4, 4)

        Credits: Implementation is inspired by that in Sophus library
                 https://github.com/strasdat/Sophus/blob/master/sophus/se3.hpp
        """
        v, omega = vec[..., :3], vec[..., 3:]
        rot_mat, theta = SO3.exp_and_theta(omega)
        theta = theta[..., None]
        s, c = np.sin(theta), np.cos(theta)

        Omega = SO3.hat(omega)
        Omega_sq = Omega @ Omega
        theta2, theta3 = theta ** 2, theta ** 3

        # Case 1: General case
        with np.errstate(divide='ignore', invalid='ignore'):
            V = np.identity(3) \
                - (c - 1.0) / theta2 * Omega \
                + (theta - s) / theta3 * Omega_sq

        # Case 2: theta ~ 0
        if np.any(theta < _EPS):
            V2 = rot_mat.data
            V = np.where(theta < _EPS, V2, V)

        trans = V @ v[..., None]
        return SE3._from_rt(rot_mat, trans)

    def log(self) -> np.ndarray:
        """Logarithm map.
        """
        raise NotImplementedError

    def transform(self, pts: np.ndarray) -> np.ndarray:
        assert len(self.shape) == pts.ndim - 2
        ptsT = pts.swapaxes(-1, -2)
        transformedT = self.data[..., :3, :3] @ ptsT + self.data[..., :3, 3:4]
        transformed = transformedT.swapaxes(-1, -2)
        return transformed

    @staticmethod
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
        return se3c.hat(v)

    @staticmethod
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
        return se3c.vee(mat)

    """Jacobians"""
    def jacob_dAexpeD_de(poseA: 'SE3', poseD: 'SE3', full_matrix: bool = True) -> np.ndarray:
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
            jac = np.zeros((*matA.shape[:-2], 12, 6), dtype=matA.dtype)
            jac[..., 9:12, 0:3] = blockA
            jac[..., 0:3, 3:6] = blockB
            jac[..., 3:6, 3:6] = blockC
            jac[..., 6:9, 3:6] = blockD
            jac[..., 9:12, 3:6] = blockE
        else:
            jac = np.concatenate([blockA, blockB, blockC, blockD, blockE], axis=-2)

        return jac

    """Comparison function"""
    def compare(self, other: 'SE3') -> Dict:
        """Compares two SO3 instances, returning the rotation error in degrees"""
        error = self * other.inv()
        e = {'rot_deg': SO3.rotation_angle(error.rot) * 180 / np.pi,
             'trans': np.linalg.norm(self.trans - other.trans, axis=-1)}
        return e

    """Conversion functions"""
    @property
    def rot(self) -> SO3:
        return SO3(self.data[..., :3, :3])

    @property
    def trans(self) -> np.array:
        return self.data[..., :3, 3]

    def vec(self) -> np.ndarray:
        """Returns the flattened representation"""
        return self.data[..., :3, :].swapaxes(-1, -2).reshape(*self.data.shape[:-2], 12)

    def as_quat_trans(self):
        """Return the 7D representation (quaternion, translation)
        First 4 columns contain the quaternion, last 3 columns contain translation
        """
        return se3c.mat2quattrans(self.data)

    def as_matrix(self) -> np.ndarray:
        return self.data

    def is_valid(self) -> bool:
        """Check whether the data is valid, e.g. if the underlying SE(3)
        representation has a valid rotation"""
        return se3c.is_valid_matrix(self.data)
