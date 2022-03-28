from typing import Union

import torch

_EPS = 1e-3  # larger epsilon for float32


def allclose(mat1: torch.Tensor, mat2: Union[torch.Tensor, float], tol=_EPS):
    """Check if all elements of two tensors are close within some tolerance.

    Either tensor can be replaced by a scalar.

    Note:
        This is similar to torch.allclose(), but considers just the absolute
        difference at a larger tolerance more suitable for float32.
    """
    return isclose(mat1, mat2, tol).all()


def isclose(mat1: torch.Tensor, mat2: Union[torch.Tensor, float], tol=_EPS):
    """Check element-wise if two tensors are close within some tolerance.

    Either tensor can be replaced by a scalar.

    Note:
        This is similar to torch.isclose(), but considers just the absolute
        difference at a larger tolerance more suitable for float32.
    """
    return (mat1 - mat2).abs_().lt(tol)