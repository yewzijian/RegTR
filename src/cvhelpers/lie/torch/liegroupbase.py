"""Generic transformation class (Torch)

References:
[1] "A tutorial on SE(3) transformation parameterizations and on-manifold
  optimization
"""
from typing import Dict, List

import torch


class LieGroupBase(object):
    DIM = None
    DOF = None
    N = None  # Group transformation is NxN matrix, e.g. 3 for SO(3)
    name = 'LieGroupBaseTorch'

    def __init__(self, data: torch.Tensor):
        """Constructor for the Lie group instance.
        Note that you should NOT call this directly, but should use one
        of the from_* methods, which will perform the appropriate checking.
        """
        self.data = data

    @staticmethod
    def identity(size: int = None, dtype=None, device=None) -> 'LieGroupBase':
        raise NotImplementedError

    @staticmethod
    def sample_uniform(size: int = None, device=None) -> 'LieGroupBase':
        raise NotImplementedError

    @staticmethod
    def from_matrix(mat: torch.Tensor, normalize: bool = False, check: bool = True) -> 'LieGroupBase':
        raise NotImplementedError

    def inv(self) -> 'LieGroupBase':
        raise NotImplementedError

    @staticmethod
    def pexp(omega: torch.Tensor) -> 'LieGroupBase':
        raise NotImplementedError

    @staticmethod
    def exp(omega: torch.Tensor) -> 'LieGroupBase':
        raise NotImplementedError

    def log(self) -> torch.Tensor:
        raise NotImplementedError

    def boxplus_left(self, delta: torch.Tensor, pseudo=False) -> 'LieGroupBase':
        """Left variant of box plus operator"""
        if pseudo:
            return self.__class__.pexp(delta) * self
        else:
            return self.__class__.exp(delta) * self

    def boxplus_right(self, delta: torch.Tensor, pseudo=False) -> 'LieGroupBase':
        """Right variant of box plus operator, i.e.
              x boxplus delta = x * exp(delta)
        See Eq (10.6) in [1]
        """
        if pseudo:
            return self * self.__class__.pexp(delta)
        else:
            return self * self.__class__.exp(delta)

    def __mul__(self, other: 'LieGroupBase') -> 'LieGroupBase':
        return self.__class__(self.data @ other.data)

    def transform(self, pts: torch.Tensor) -> torch.Tensor:
        """Applies the transformation on points

        Args:
            pts: Points to transform. Should have the size [N, N_pts, 3] if
              transform is batched else, [N_pts, 3]
        """
        raise NotImplementedError

    def compare(self, other: 'LieGroupBase') -> Dict:
        """Compare with another instance"""
        raise NotImplementedError

    def vec(self) -> torch.Tensor:
        """Returns the flattened representation"""
        raise NotImplementedError

    def as_matrix(self) -> torch.Tensor:
        """Return the matrix form of the transform (e.g. 3x3 for SO(3))"""
        return self.data

    def is_valid(self) -> bool:
        """Check whether the data is valid, e.g. if the underlying SE(3)
        representation has a valid rotation"""
        raise NotImplementedError

    def make_valid(self):
        """Rectifies the data so that the representation is valid"""
        pass

    """Misc methods"""
    def __getitem__(self, item) -> 'LieGroupBase':
        return self.__class__(self.data[item])

    def __setitem__(self, key, value):
        if isinstance(value, torch.Tensor):
            self.data[key] = value
        else:
            self.data[key] = value.data

    def __repr__(self):
        return '{} containing {}'.format(self.name, str(self.data))

    def __str__(self):
        return '{}{}'.format(self.name, list(self.data.shape[:-2]))

    @property
    def shape(self):
        return self.data.shape[:-2]

    def __len__(self):
        shape = self.shape
        return self.shape[0] if len(shape) >= 1 else 1

    @classmethod
    def stack(cls, transforms: List['LieGroupBase']):
        """Concatenates transforms into a single transform"""
        stacked = torch.cat([t.data for t in transforms], dim=0)
        return cls(stacked)

    """Torch specific methods"""
    def to(self, device) -> 'LieGroupBase':
        """Move instance to device"""
        self.data = self.data.to(device)
        return self

    def type(self, dtype) -> 'LieGroupBase':
        self.data = self.data.type(dtype)
        return self

    def detach(self) -> 'LieGroupBase':
        return self.__class__(self.data.detach())

