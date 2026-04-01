"""Lloyd-Max optimal scalar quantizer for Gaussian distributions.

Precomputes codebooks (centroids + boundaries) for N(0,1) at various bit widths.
After random orthogonal rotation, each coordinate of a high-dimensional vector
follows approximately N(0, sigma) — so we quantize the normalized (unit-variance)
version and scale by the norm.

The codebooks are data-oblivious: same for any model, any layer, any head.
"""

import math

import mlx.core as mx
import numpy as np


def _lloyd_max_gaussian(n_levels: int, max_iter: int = 200, tol: float = 1e-8):
    """Compute Lloyd-Max optimal quantizer for N(0,1).

    Returns (boundaries, centroids) as numpy arrays.
    - boundaries: (n_levels + 1,) including -inf and +inf
    - centroids: (n_levels,) reconstruction values
    """
    # Initialize centroids uniformly in [-3, 3] (covers 99.7% of N(0,1))
    centroids = np.linspace(-3.0, 3.0, n_levels)

    for _ in range(max_iter):
        # Boundaries are midpoints between consecutive centroids
        boundaries = np.empty(n_levels + 1)
        boundaries[0] = -np.inf
        boundaries[-1] = np.inf
        for i in range(1, n_levels):
            boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0

        # Update centroids: E[X | boundary[i] < X < boundary[i+1]] for N(0,1)
        new_centroids = np.empty(n_levels)
        for i in range(n_levels):
            lo, hi = boundaries[i], boundaries[i + 1]
            # For Gaussian: E[X | lo < X < hi] = (phi(lo) - phi(hi)) / (Phi(hi) - Phi(lo))
            # where phi = pdf, Phi = cdf
            phi_lo = _gaussian_pdf(lo)
            phi_hi = _gaussian_pdf(hi)
            cdf_lo = _gaussian_cdf(lo)
            cdf_hi = _gaussian_cdf(hi)
            prob = cdf_hi - cdf_lo
            if prob < 1e-15:
                new_centroids[i] = (lo + hi) / 2.0 if np.isfinite(lo) and np.isfinite(hi) else centroids[i]
            else:
                new_centroids[i] = (phi_lo - phi_hi) / prob

        if np.max(np.abs(new_centroids - centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    # Recompute final boundaries
    boundaries = np.empty(n_levels + 1)
    boundaries[0] = -np.inf
    boundaries[-1] = np.inf
    for i in range(1, n_levels):
        boundaries[i] = (centroids[i - 1] + centroids[i]) / 2.0

    return boundaries, centroids


def _gaussian_pdf(x):
    """Standard normal PDF."""
    if not np.isfinite(x):
        return 0.0
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _gaussian_cdf(x):
    """Standard normal CDF."""
    if x == -np.inf:
        return 0.0
    if x == np.inf:
        return 1.0
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# Precompute codebooks for common bit widths
_CODEBOOKS: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def get_codebook(bits: int) -> tuple[mx.array, mx.array]:
    """Get (boundaries, centroids) for the given bit width as MLX arrays.

    Cached after first computation. boundaries shape: (2^bits + 1,),
    centroids shape: (2^bits,).
    """
    if bits not in _CODEBOOKS:
        n_levels = 1 << bits
        boundaries, centroids = _lloyd_max_gaussian(n_levels)
        _CODEBOOKS[bits] = (boundaries, centroids)

    boundaries, centroids = _CODEBOOKS[bits]
    return (
        mx.array(boundaries, dtype=mx.float32),
        mx.array(centroids, dtype=mx.float32),
    )


def quantize_scalar(x: mx.array, bits: int) -> tuple[mx.array, mx.array, mx.array]:
    """Quantize a float array using Lloyd-Max optimal quantizer.

    Args:
        x: Input tensor of any shape.
        bits: Number of bits per element (2, 3, 4, 6, 8).

    Returns:
        (indices, norms, codebook_centroids):
        - indices: uint8/uint16 array of quantization indices, same shape as x
        - norms: per-vector L2 norms (shape depends on caller's grouping)
        - codebook_centroids: the centroids array for dequantization
    """
    boundaries, centroids = get_codebook(bits)

    # Find which bin each value falls into
    # boundaries[1:-1] are the finite decision boundaries
    finite_bounds = boundaries[1:-1]  # shape: (2^bits - 1,)

    # Nearest-centroid assignment: find the centroid closest to each value
    # This is equivalent to searchsorted on boundaries but works in MLX
    # Expand dims for broadcasting: x[..., None] vs centroids[None, ...]
    x_expanded = x[..., None]  # (..., n_levels)
    dists = mx.abs(x_expanded - centroids)  # broadcast: (..., n_levels)
    indices = mx.argmin(dists, axis=-1)

    return indices, centroids


def dequantize_scalar(indices: mx.array, centroids: mx.array) -> mx.array:
    """Dequantize indices back to float values using codebook centroids.

    Args:
        indices: Integer indices into the codebook.
        centroids: Codebook centroids from quantize_scalar.

    Returns:
        Reconstructed float values, same shape as indices.
    """
    return centroids[indices]
