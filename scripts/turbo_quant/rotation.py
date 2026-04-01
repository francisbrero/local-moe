"""Random orthogonal rotation matrices for TurboQuant.

Generates a fixed rotation matrix R via QR decomposition of a random Gaussian
matrix. The same seed produces the same R for reproducibility. R is orthogonal
(R @ R^T = I), so it preserves norms and inner products.

After rotation, coordinates of high-dimensional vectors become approximately
i.i.d. Gaussian, which is ideal for the Lloyd-Max scalar quantizer.
"""

import mlx.core as mx
import numpy as np


def make_rotation_matrix(dim: int, seed: int = 42, dtype=mx.float16) -> mx.array:
    """Generate a random orthogonal matrix of shape (dim, dim).

    Uses QR decomposition of a random Gaussian matrix (Haar-distributed
    orthogonal matrix). The seed ensures reproducibility across runs.

    Args:
        dim: Dimension of the rotation matrix (typically head_dim).
        seed: Random seed for reproducibility.
        dtype: Output dtype (float16 for memory efficiency).

    Returns:
        Orthogonal matrix R of shape (dim, dim) as MLX array.
    """
    rng = np.random.RandomState(seed)
    G = rng.randn(dim, dim).astype(np.float32)
    Q, R_upper = np.linalg.qr(G)
    # Ensure deterministic sign (Haar measure convention)
    signs = np.sign(np.diag(R_upper))
    signs[signs == 0] = 1.0
    Q = Q * signs[np.newaxis, :]
    return mx.array(Q, dtype=dtype)


def rotate(x: mx.array, R: mx.array) -> mx.array:
    """Apply rotation: x_rotated = x @ R.

    Args:
        x: Input tensor of shape (..., dim).
        R: Rotation matrix of shape (dim, dim).

    Returns:
        Rotated tensor of shape (..., dim).
    """
    return x @ R


def inverse_rotate(x: mx.array, R: mx.array) -> mx.array:
    """Apply inverse rotation: x_original = x @ R^T.

    Since R is orthogonal, R^{-1} = R^T.

    Args:
        x: Rotated tensor of shape (..., dim).
        R: Rotation matrix of shape (dim, dim).

    Returns:
        Original-space tensor of shape (..., dim).
    """
    return x @ R.T
