"""Compare TurboQuant (rotation + Lloyd-Max) vs MLX-LM built-in (group quantization).

Measures attention fidelity (cosine similarity), compression ratio, and quantization
overhead for both approaches on the same synthetic data.
"""

import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_utils import get_environment_info, log_experiment
from turbo_quant.core import (
    TurboQuantConfig,
    compressed_attention,
    fp16_kv_bytes,
    kv_bytes,
    quantize_kv,
)


def cosine_similarity(a: mx.array, b: mx.array) -> float:
    a_flat = a.reshape(-1).astype(mx.float32)
    b_flat = b.reshape(-1).astype(mx.float32)
    dot = mx.sum(a_flat * b_flat)
    norm_a = mx.linalg.norm(a_flat)
    norm_b = mx.linalg.norm(b_flat)
    sim = dot / (norm_a * norm_b + 1e-8)
    mx.eval(sim)
    return float(sim.item())


def reference_attention(queries, keys, values, mask=None):
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


def builtin_quantize_kv(keys, values, bits=4, group_size=64):
    """Use MLX's built-in mx.quantize for group quantization."""
    batch, n_heads, seq_len, head_dim = keys.shape

    # MLX quantize works on 2D arrays, reshape for it
    k_flat = keys.reshape(-1, head_dim)
    v_flat = values.reshape(-1, head_dim)

    k_quant, k_scales, k_biases = mx.quantize(k_flat, bits=bits, group_size=group_size)
    v_quant, v_scales, v_biases = mx.quantize(v_flat, bits=bits, group_size=group_size)

    return {
        "k_quant": k_quant, "k_scales": k_scales, "k_biases": k_biases,
        "v_quant": v_quant, "v_scales": v_scales, "v_biases": v_biases,
        "shape": (batch, n_heads, seq_len, head_dim),
        "bits": bits, "group_size": group_size,
    }


def builtin_dequantize_kv(quantized):
    batch, n_heads, seq_len, head_dim = quantized["shape"]
    k_deq = mx.dequantize(
        quantized["k_quant"], quantized["k_scales"], quantized["k_biases"],
        bits=quantized["bits"], group_size=quantized["group_size"],
    ).reshape(batch, n_heads, seq_len, head_dim)
    v_deq = mx.dequantize(
        quantized["v_quant"], quantized["v_scales"], quantized["v_biases"],
        bits=quantized["bits"], group_size=quantized["group_size"],
    ).reshape(batch, n_heads, seq_len, head_dim)
    return k_deq, v_deq


def builtin_kv_bytes(quantized):
    total = 0
    for key in ["k_quant", "k_scales", "k_biases", "v_quant", "v_scales", "v_biases"]:
        total += quantized[key].nbytes
    return total


def compare_methods(
    n_kv_heads=4, n_q_heads=16, head_dim=128, seq_len=2048, bits=4,
):
    """Compare TurboQuant vs built-in on identical data."""
    batch = 1
    queries = mx.random.normal(shape=(batch, n_q_heads, 1, head_dim)).astype(mx.float16)
    keys = mx.random.normal(shape=(batch, n_kv_heads, seq_len, head_dim)).astype(mx.float16)
    values = mx.random.normal(shape=(batch, n_kv_heads, seq_len, head_dim)).astype(mx.float16)
    mx.eval(queries, keys, values)

    # Reference FP16 attention
    ref_output = reference_attention(queries, keys, values)
    mx.eval(ref_output)

    fp16_bytes = fp16_kv_bytes(batch, n_kv_heads, seq_len, head_dim)

    results = {}

    # --- TurboQuant (rotation + Lloyd-Max) ---
    config = TurboQuantConfig(head_dim=head_dim, k_bits=bits, v_bits=bits)
    t0 = time.perf_counter()
    tq_quantized = quantize_kv(keys, values, config)
    mx.eval(*tq_quantized.values())
    tq_quant_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    tq_output = compressed_attention(queries, tq_quantized, config)
    mx.eval(tq_output)
    tq_attn_ms = (time.perf_counter() - t0) * 1000

    tq_cos = cosine_similarity(ref_output, tq_output)
    tq_bytes = kv_bytes(tq_quantized)
    tq_ratio = fp16_bytes / tq_bytes

    results["turbo_quant"] = {
        "cosine": tq_cos,
        "compressed_bytes": tq_bytes,
        "ratio": round(tq_ratio, 3),
        "quant_ms": round(tq_quant_ms, 2),
        "attn_ms": round(tq_attn_ms, 2),
    }

    # --- Built-in (mx.quantize group quant) ---
    t0 = time.perf_counter()
    bi_quantized = builtin_quantize_kv(keys, values, bits=bits)
    mx.eval(bi_quantized["k_quant"], bi_quantized["v_quant"])
    bi_quant_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    k_deq, v_deq = builtin_dequantize_kv(bi_quantized)
    bi_output = reference_attention(queries, k_deq, v_deq)
    mx.eval(bi_output)
    bi_attn_ms = (time.perf_counter() - t0) * 1000

    bi_cos = cosine_similarity(ref_output, bi_output)
    bi_bytes = builtin_kv_bytes(bi_quantized)
    bi_ratio = fp16_bytes / bi_bytes

    results["builtin"] = {
        "cosine": bi_cos,
        "compressed_bytes": bi_bytes,
        "ratio": round(bi_ratio, 3),
        "quant_ms": round(bi_quant_ms, 2),
        "attn_ms": round(bi_attn_ms, 2),
    }

    return results


# Model configs
MODEL_CONFIGS = {
    "S": {"name": "Qwen2.5-0.5B", "n_kv_heads": 2, "head_dim": 64, "n_q_heads": 12, "n_layers": 24},
    "M": {"name": "Qwen2.5-3B", "n_kv_heads": 2, "head_dim": 128, "n_q_heads": 16, "n_layers": 36},
    "XL": {"name": "Qwen2.5-14B", "n_kv_heads": 4, "head_dim": 128, "n_q_heads": 40, "n_layers": 48},
}


def main():
    env = get_environment_info()
    print("=== TurboQuant vs Built-in KV Cache Quantization ===\n")

    for tier, cfg in MODEL_CONFIGS.items():
        print(f"--- {tier}: {cfg['name']} ({cfg['n_kv_heads']} KV heads, dim={cfg['head_dim']}) ---")

        for bits in [4, 8]:
            for seq_len in [512, 2048, 8192]:
                results = compare_methods(
                    n_kv_heads=cfg["n_kv_heads"],
                    n_q_heads=cfg["n_q_heads"],
                    head_dim=cfg["head_dim"],
                    seq_len=seq_len,
                    bits=bits,
                )

                tq = results["turbo_quant"]
                bi = results["builtin"]

                print(f"\n  {bits}-bit, seq_len={seq_len}:")
                print(f"    {'Method':>14} {'Cosine':>10} {'Ratio':>8} {'Quant ms':>10} {'Attn ms':>10}")
                print(f"    {'-'*50}")
                print(f"    {'TurboQuant':>14} {tq['cosine']:>10.6f} {tq['ratio']:>8.2f}x {tq['quant_ms']:>10.1f} {tq['attn_ms']:>10.1f}")
                print(f"    {'Built-in':>14} {bi['cosine']:>10.6f} {bi['ratio']:>8.2f}x {bi['quant_ms']:>10.1f} {bi['attn_ms']:>10.1f}")

                winner = "TurboQuant" if tq["cosine"] > bi["cosine"] else "Built-in"
                delta = abs(tq["cosine"] - bi["cosine"])
                print(f"    Winner: {winner} (Δ={delta:.6f})")

                log_experiment(
                    experiment_name=f"turbo_quant_comparison_{tier.lower()}_{bits}bit_{seq_len}",
                    phase="turbo_quant_comparison",
                    config={
                        "tier": tier,
                        "model": cfg["name"],
                        "bits": bits,
                        "seq_len": seq_len,
                        "n_kv_heads": cfg["n_kv_heads"],
                        "head_dim": cfg["head_dim"],
                    },
                    results=results,
                    env=env,
                )

        print()

    print("=== Done ===")


if __name__ == "__main__":
    main()
