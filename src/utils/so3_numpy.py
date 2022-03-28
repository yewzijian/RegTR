"""Functions for performing operations related to rigid_transformations (Numpy).

Note that poses are stored in 3x4 matrices, i.e. the last row isn't stored.
Unlike otherwise stated, all functions support arbitrary batch dimensions, e.g.
poses can have shapes ([N,] 3, 4)
"""

import numpy as np


def so3_transform(rot, xyz):
    """

    Args:
        rot: ([B,] 3, 3)
        xyz: ([B,] N, 3)

    Returns:

    """
    assert xyz.shape[-1] == 3 and rot.shape[:-2] == xyz.shape[:-2]
    transformed = np.einsum('...ij,...bj->...bi', rot, xyz)
    return transformed
