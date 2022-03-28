from typing import Dict, List

import numpy as np

_EPS = 1e-5


# Generic transformation class (Numpy)
class LieGroupBase(object):

    DIM = None
    DOF = None
    N = None  # Group transformation is NxN matrix, e.g. 3 for SO(3)
    name = 'LieGroupBaseNumpy'

    def __init__(self, data: np.ndarray):
        """Constructor for the Lie group instance.
        Note that you should NOT call this directly, but should use one
        of the from_* methods, which will perform the appropriate checking.
        """
        self.data = data

    @staticmethod
    def identity(size: int = None) -> 'LieGroupBase':
        raise NotImplementedError

    @staticmethod
    def sample_uniform(size: int = 1) -> 'LieGroupBase':
        raise NotImplementedError

    @staticmethod
    def sample_small(size: int = None, std=None) -> 'LieGroupBase':
        raise NotImplementedError

    @staticmethod
    def from_matrix(mat: np.ndarray, normalize: bool = False, check: bool = True) -> 'LieGroupBase':
        raise NotImplementedError

    @staticmethod
    def exp(omega: np.ndarray) -> 'LieGroupBase':
        raise NotImplementedError

    def log(self) -> np.ndarray:
        raise NotImplementedError

    def boxplus_left(self, delta: np.ndarray) -> 'LieGroupBase':
        """Left variant of box plus operator"""
        return self.__class__.exp(delta) * self

    def boxplus_right(self, delta: np.ndarray) -> 'LieGroupBase':
        """Right variant of box plus operator, i.e.
              x boxplus delta = x * exp(delta)
        See Eq (10.6) in [1]
        """
        return self * self.__class__.exp(delta)

    def inv(self) -> 'LieGroupBase':
        raise NotImplementedError

    def __mul__(self, other: 'LieGroupBase') -> 'LieGroupBase':
        return self.__class__(self.data @ other.data)

    def transform(self, pts: np.ndarray) -> np.ndarray:
        """Applies the transformation on points

        Args:
            pts: Points to transform. Should have the size [N, N_pts, 3] if
              transform is batched else, [N_pts, 3]
        """
        raise NotImplementedError

    def compare(self, other: 'LieGroupBase') -> Dict:
        """Compare with another instance"""
        raise NotImplementedError

    def vec(self) -> np.ndarray:
        """Returns the flattened representation"""
        raise NotImplementedError

    def as_matrix(self) -> np.ndarray:
        """Return the matrix form of the transform (e.g. 3x3 for SO(3))"""
        return self.data

    def is_valid(self) -> bool:
        """Check whether the data is valid, e.g. if the underlying SE(3)
        representation has a valid rotation"""
        raise NotImplementedError

    """Misc Methods"""
    def __getitem__(self, item) -> 'LieGroupBase':
        return self.__class__(self.data[item])

    def __setitem__(self, index, value):
        self.data[index] = np.array(value)

    def __repr__(self):
        return '{} containing {}'.format(self.name, str(self.data))

    def __str__(self):
        return '{}{}'.format(self.name, list(self.data.shape[:-2]))

    def __array__(self):
        return self.data

    @property
    def shape(self):
        return self.data.shape[:-2]

    def __len__(self):
        shape = self.shape
        return self.shape[0] if len(shape) >= 1 else 1

    @classmethod
    def stack(cls, transforms: List['LieGroupBase']):
        """Concatenates transforms into a single transform"""
        stacked = np.concatenate([t.data for t in transforms], axis=0)
        return cls(stacked)
