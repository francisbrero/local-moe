"""Accuracy validation for TurboQuant KV cache compression.

Tests:
1. Attention fidelity: cosine similarity and L2 error vs FP16
2. Roundtrip reconstruction quality
3. Inner product preservation (rotated-space correctness proof)
4. Compression ratio verification via direct byte accounting
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from turbo_quant.core import (
    TurboQuantConfig,
    compressed_attention,
    fp16_kv_bytes,
    kv_bytes,
    quantize_kv,
)


def cosine_similarity(a: mx.array, b: mx.array) -> float:
    """Compute cosine similarity between two tensors (flattened)."""
    a_flat = a.reshape(-1).astype(mx.float32)
    b_flat = b.reshape(-1).astype(mx.float32)
    dot = mx.sum(a_flat * b_flat)
    norm_a = mx.linalg.norm(a_flat)
    norm_b = mx.linalg.norm(b_flat)
    sim = dot / (norm_a * norm_b + 1e-8)
    mx.eval(sim)
    return float(sim.item())


def l2_error(a: mx.array, b: mx.array) -> float:
    """Compute relative L2 error."""
    a_flat = a.reshape(-1).astype(mx.float32)
    b_flat = b.reshape(-1).astype(mx.float32)
    err = mx.linalg.norm(a_flat - b_flat)
    ref = mx.linalg.norm(a_flat)
    rel = err / (ref + 1e-8)
    mx.eval(rel)
    return float(rel.item())


def reference_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    mask: mx.array | None = None,
) -> mx.array:
    """Standard FP16 attention for comparison."""
    head_dim = queries.shape[-1]
    n_q_heads = queries.shape[1]
    n_kv_heads = keys.shape[1]
    scale = head_dim ** -0.5

    if n_kv_heads < n_q_heads:
        repeats = n_q_heads // n_kv_heads
        keys = mx.repeat(keys, repeats, axis=1)
        values = mx.repeat(values, repeats, axis=1)

    scores = (queries @ keys.transpose(0, 1, 3, 2)) * scale
    if mask is not None:
        scores = scores + mask
    weights = mx.softmax(scores, axis=-1).astype(mx.float16)
    return weights @ values


def test_roundtrip_reconstruction(head_dim: int = 128, bits: int = 4):
    """Test that quantize -> dequantize preserves vectors approximately."""
    from turbo_quant.core import dequantize_keys, dequantize_values
    from turbo_quant.rotation import inverse_rotate

    config = TurboQuantConfig(head_dim=head_dim, k_bits=bits, v_bits=bits)
    batch, n_heads, seq_len = 1, 4, 512
    keys = mx.random.normal(shape=(batch, n_heads, seq_len, head_dim)).astype(mx.float16)
    values = mx.random.normal(shape=(batch, n_heads, seq_len, head_dim)).astype(mx.float16)

    quantized = quantize_kv(keys, values, config)

    # Dequantize and inverse-rotate to get back to original space
    k_rot = dequantize_keys(quantized, config)
    v_rot = dequantize_values(quantized, config)
    k_recon = inverse_rotate(k_rot, config.R)
    v_recon = inverse_rotate(v_rot, config.R)

    mx.eval(k_recon, v_recon)

    k_cos = cosine_similarity(keys, k_recon)
    v_cos = cosine_similarity(values, v_recon)
    k_l2 = l2_error(keys, k_recon)
    v_l2 = l2_error(values, v_recon)

    print(f"  Roundtrip ({bits}-bit, dim={head_dim}):")
    print(f"    K cosine={k_cos:.6f}, L2_rel={k_l2:.6f}")
    print(f"    V cosine={v_cos:.6f}, L2_rel={v_l2:.6f}")
    return k_cos, v_cos


def test_inner_product_preservation(head_dim: int = 128, bits: int = 4):
    """Test that Q @ K^T ≈ Q_rot @ K_rot^T (exact for unquantized, approximate for quantized)."""
    from turbo_quant.rotation import make_rotation_matrix, rotate

    config = TurboQuantConfig(head_dim=head_dim, k_bits=bits, v_bits=bits)
    batch, n_heads, seq_len = 1, 4, 256
    queries = mx.random.normal(shape=(batch, n_heads, 1, head_dim)).astype(mx.float16)
    keys = mx.random.normal(shape=(batch, n_heads, seq_len, head_dim)).astype(mx.float16)

    # Reference: Q @ K^T
    ref_scores = queries @ keys.transpose(0, 1, 3, 2)

    # Quantized path: Q_rot @ K_rot_quantized^T
    from turbo_quant.core import dequantize_keys

    quantized = quantize_kv(keys, keys, config)  # values not used here
    k_rot = dequantize_keys(quantized, config)
    q_rot = rotate(queries, config.R)
    quant_scores = q_rot @ k_rot.transpose(0, 1, 3, 2)

    mx.eval(ref_scores, quant_scores)

    cos = cosine_similarity(ref_scores, quant_scores)
    l2 = l2_error(ref_scores, quant_scores)

    print(f"  Inner product preservation ({bits}-bit):")
    print(f"    Score cosine={cos:.6f}, L2_rel={l2:.6f}")
    return cos


def test_attention_fidelity(
    head_dim: int = 128,
    n_kv_heads: int = 4,
    n_q_heads: int = 16,
    seq_len: int = 512,
    bits: int = 4,
):
    """Test full attention output: compressed vs FP16 reference."""
    config = TurboQuantConfig(head_dim=head_dim, k_bits=bits, v_bits=bits)
    batch = 1
    queries = mx.random.normal(shape=(batch, n_q_heads, 1, head_dim)).astype(mx.float16)
    keys = mx.random.normal(shape=(batch, n_kv_heads, seq_len, head_dim)).astype(mx.float16)
    values = mx.random.normal(shape=(batch, n_kv_heads, seq_len, head_dim)).astype(mx.float16)

    # Causal mask
    mask = mx.full((1, seq_len), 0.0, dtype=mx.float16).reshape(1, 1, 1, seq_len)

    # Reference
    ref_output = reference_attention(queries, keys, values, mask)
    mx.eval(ref_output)

    # Compressed
    quantized = quantize_kv(keys, values, config)
    comp_output = compressed_attention(queries, quantized, config, mask)
    mx.eval(comp_output)

    cos = cosine_similarity(ref_output, comp_output)
    l2 = l2_error(ref_output, comp_output)

    # Byte accounting
    comp_bytes = kv_bytes(quantized)
    fp16_bytes = fp16_kv_bytes(batch, n_kv_heads, seq_len, head_dim)
    ratio = fp16_bytes / comp_bytes

    print(f"  Attention fidelity ({bits}-bit, seq={seq_len}, GQA {n_q_heads}q/{n_kv_heads}kv):")
    print(f"    Output cosine={cos:.6f}, L2_rel={l2:.6f}")
    print(f"    Compression: {comp_bytes:,} bytes vs {fp16_bytes:,} FP16 = {ratio:.2f}x")
    return cos, ratio


def test_context_length_scaling(bits: int = 4):
    """Test attention fidelity across increasing context lengths."""
    print(f"\n  Context length scaling ({bits}-bit):")
    head_dim = 128
    n_kv_heads = 4
    n_q_heads = 16

    results = []
    for seq_len in [128, 512, 1024, 2048, 4096]:
        config = TurboQuantConfig(head_dim=head_dim, k_bits=bits, v_bits=bits)
        batch = 1
        queries = mx.random.normal(shape=(batch, n_q_heads, 1, head_dim)).astype(mx.float16)
        keys = mx.random.normal(shape=(batch, n_kv_heads, seq_len, head_dim)).astype(mx.float16)
        values = mx.random.normal(shape=(batch, n_kv_heads, seq_len, head_dim)).astype(mx.float16)

        ref_output = reference_attention(queries, keys, values)
        mx.eval(ref_output)

        quantized = quantize_kv(keys, values, config)
        comp_output = compressed_attention(queries, quantized, config)
        mx.eval(comp_output)

        cos = cosine_similarity(ref_output, comp_output)
        l2 = l2_error(ref_output, comp_output)
        print(f"    seq_len={seq_len:5d}: cosine={cos:.6f}, L2_rel={l2:.6f}")
        results.append((seq_len, cos, l2))

    return results


def main():
    print("=== TurboQuant Accuracy Validation ===\n")

    print("1. Roundtrip Reconstruction:")
    for bits in [4, 3]:
        k_cos, v_cos = test_roundtrip_reconstruction(bits=bits)
        status = "PASS" if min(k_cos, v_cos) > 0.99 else "FAIL"
        print(f"    [{status}] {bits}-bit: min cosine = {min(k_cos, v_cos):.6f}")

    print("\n2. Inner Product Preservation:")
    for bits in [4, 3]:
        cos = test_inner_product_preservation(bits=bits)
        status = "PASS" if cos > 0.99 else "FAIL"
        print(f"    [{status}] {bits}-bit: score cosine = {cos:.6f}")

    print("\n3. Attention Fidelity:")
    for bits in [4, 3]:
        cos, ratio = test_attention_fidelity(bits=bits)
        target = 0.999 if bits == 4 else 0.995
        status = "PASS" if cos > target else "FAIL"
        print(f"    [{status}] {bits}-bit: cosine={cos:.6f} (target>{target}), compression={ratio:.2f}x")

    print("\n4. Context Length Scaling:")
    for bits in [4, 3]:
        results = test_context_length_scaling(bits=bits)
        # Check for error growth
        cosines = [r[1] for r in results]
        if len(cosines) >= 2:
            drift = cosines[0] - cosines[-1]
            print(f"    Drift (short→long): {drift:.6f} ({'STABLE' if abs(drift) < 0.005 else 'DEGRADING'})")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
