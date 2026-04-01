"""Core TurboQuant quantize/dequantize for KV cache vectors.

Design: operate in rotated space.
- Keys: stored rotated + quantized. Queries rotated once per token.
  Q_rot @ K_rot^T = (Q @ R) @ (K @ R)^T = Q @ R @ R^T @ K^T = Q @ K^T.
- Values: stored rotated + quantized. Attention output inverse-rotated once per token.
  output_rot = softmax(scores) @ V_rot, then output = output_rot @ R^T.

Both operations are O(d^2) per token, independent of sequence length.
"""

import mlx.core as mx

from .lloyd_max import dequantize_scalar, get_codebook, quantize_scalar
from .rotation import inverse_rotate, make_rotation_matrix, rotate


class TurboQuantConfig:
    """Configuration for TurboQuant compression."""

    def __init__(
        self,
        head_dim: int,
        k_bits: int = 4,
        v_bits: int = 4,
        seed: int = 42,
    ):
        self.head_dim = head_dim
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.seed = seed
        # One rotation matrix shared by all heads (same dim)
        self.R = make_rotation_matrix(head_dim, seed=seed)
        # Preload codebooks
        self._k_centroids = get_codebook(k_bits)[1]
        self._v_centroids = get_codebook(v_bits)[1]


def quantize_kv(
    keys: mx.array,
    values: mx.array,
    config: TurboQuantConfig,
) -> dict:
    """Quantize K and V tensors for cache storage.

    Args:
        keys: shape (batch, n_kv_heads, seq_len, head_dim), float16
        values: shape (batch, n_kv_heads, seq_len, head_dim), float16
        config: TurboQuantConfig

    Returns:
        dict with:
        - k_indices: uint8 array of quantized key indices
        - k_norms: float32 per-vector norms for keys
        - v_indices: uint8 array of quantized value indices
        - v_norms: float32 per-vector norms for values
    """
    R = config.R

    # --- Keys ---
    # Rotate first. After rotation of a high-dim vector, coordinates become
    # approximately i.i.d. N(0, sigma) where sigma = ||x|| / sqrt(d).
    # We normalize by per-vector std to get ~N(0,1), matching the Lloyd-Max codebook.
    k_rotated = rotate(keys, R).astype(mx.float32)
    k_std = mx.sqrt(mx.mean(k_rotated * k_rotated, axis=-1, keepdims=True))
    k_std = mx.maximum(k_std, 1e-8)
    k_normalized = k_rotated / k_std
    k_indices, _ = quantize_scalar(k_normalized, config.k_bits)
    k_indices = k_indices.astype(mx.uint8) if config.k_bits <= 8 else k_indices.astype(mx.uint16)

    # --- Values ---
    v_rotated = rotate(values, R).astype(mx.float32)
    v_std = mx.sqrt(mx.mean(v_rotated * v_rotated, axis=-1, keepdims=True))
    v_std = mx.maximum(v_std, 1e-8)
    v_normalized = v_rotated / v_std
    v_indices, _ = quantize_scalar(v_normalized, config.v_bits)
    v_indices = v_indices.astype(mx.uint8) if config.v_bits <= 8 else v_indices.astype(mx.uint16)

    return {
        "k_indices": k_indices,
        "k_scales": k_std.squeeze(-1).astype(mx.float16),
        "v_indices": v_indices,
        "v_scales": v_std.squeeze(-1).astype(mx.float16),
    }


def dequantize_keys(quantized: dict, config: TurboQuantConfig) -> mx.array:
    """Dequantize keys back to rotated space (NOT original space).

    Returns keys in rotated space for dot product with rotated queries.
    Shape: same as original keys.
    """
    k_recon = dequantize_scalar(quantized["k_indices"], config._k_centroids)
    # Undo the std normalization
    k_scales = quantized["k_scales"].astype(mx.float32)[..., None]
    return (k_recon * k_scales).astype(mx.float16)


def dequantize_values(quantized: dict, config: TurboQuantConfig) -> mx.array:
    """Dequantize values back to rotated space.

    Returns values in rotated space. The attention output must be
    inverse-rotated after the weighted sum.
    Shape: same as original values.
    """
    v_recon = dequantize_scalar(quantized["v_indices"], config._v_centroids)
    v_scales = quantized["v_scales"].astype(mx.float32)[..., None]
    return (v_recon * v_scales).astype(mx.float16)


def compressed_attention(
    queries: mx.array,
    quantized_cache: dict,
    config: TurboQuantConfig,
    mask: mx.array | None = None,
) -> mx.array:
    """Compute attention with compressed KV cache.

    Args:
        queries: (batch, n_heads, q_len, head_dim), float16
        quantized_cache: dict from quantize_kv
        config: TurboQuantConfig
        mask: optional attention mask

    Returns:
        Attention output in original (unrotated) space, same shape as queries.
    """
    R = config.R
    head_dim = config.head_dim
    scale = head_dim ** -0.5

    # Rotate queries once: O(q_len * d^2)
    q_rot = rotate(queries, R)

    # Dequantize K and V in rotated space
    k_rot = dequantize_keys(quantized_cache, config)
    v_rot = dequantize_values(quantized_cache, config)

    # Handle GQA: expand k/v heads if fewer than query heads
    n_q_heads = queries.shape[1]
    n_kv_heads = k_rot.shape[1]
    if n_kv_heads < n_q_heads:
        repeats = n_q_heads // n_kv_heads
        k_rot = mx.repeat(k_rot, repeats, axis=1)
        v_rot = mx.repeat(v_rot, repeats, axis=1)

    # Attention scores in rotated space: Q_rot @ K_rot^T = Q @ K^T (exact)
    scores = (q_rot @ k_rot.transpose(0, 1, 3, 2)) * scale

    if mask is not None:
        scores = scores + mask

    weights = mx.softmax(scores, axis=-1).astype(mx.float16)

    # Weighted sum in rotated space
    output_rot = weights @ v_rot

    # Inverse rotate output once: O(q_len * d^2)
    output = inverse_rotate(output_rot, R)

    return output


def kv_bytes(quantized: dict) -> int:
    """Compute exact storage used by compressed KV cache in bytes.

    Includes indices, norms, and any metadata. Does NOT include the
    rotation matrix or codebook (these are shared constants).
    """
    total = 0
    for key, arr in quantized.items():
        total += arr.nbytes
    return total


def fp16_kv_bytes(batch: int, n_kv_heads: int, seq_len: int, head_dim: int) -> int:
    """Compute equivalent FP16 KV cache size in bytes for comparison."""
    # 2 (K+V) * batch * n_kv_heads * seq_len * head_dim * 2 (float16)
    return 2 * batch * n_kv_heads * seq_len * head_dim * 2
