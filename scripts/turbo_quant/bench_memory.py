"""Memory savings benchmark for TurboQuant KV cache compression.

Measures:
- Direct byte accounting via kv_bytes() (ground truth for compression ratio)
- Per-token KV storage cost from cache growth measurement
- Peak RSS and Metal GPU memory as supporting metrics
- Compression ratio at various context lengths
"""

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_utils import get_environment_info, log_experiment
from turbo_quant.core import (
    TurboQuantConfig,
    fp16_kv_bytes,
    kv_bytes,
    quantize_kv,
)


def measure_compression(
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
    k_bits: int = 4,
    v_bits: int = 4,
    batch: int = 1,
) -> dict:
    """Measure compression ratio for given configuration."""
    config = TurboQuantConfig(head_dim=head_dim, k_bits=k_bits, v_bits=v_bits)

    # Generate synthetic KV tensors
    keys = mx.random.normal(shape=(batch, n_kv_heads, seq_len, head_dim)).astype(mx.float16)
    values = mx.random.normal(shape=(batch, n_kv_heads, seq_len, head_dim)).astype(mx.float16)
    mx.eval(keys, values)

    # Quantize
    t0 = time.perf_counter()
    quantized = quantize_kv(keys, values, config)
    mx.eval(*quantized.values())
    quant_time = time.perf_counter() - t0

    # Measure bytes
    comp_bytes = kv_bytes(quantized)
    fp16_bytes_val = fp16_kv_bytes(batch, n_kv_heads, seq_len, head_dim)
    ratio = fp16_bytes_val / comp_bytes

    # Per-token cost
    per_token_bytes = comp_bytes / seq_len
    per_token_fp16 = fp16_bytes_val / seq_len

    return {
        "seq_len": seq_len,
        "compressed_bytes": comp_bytes,
        "fp16_bytes": fp16_bytes_val,
        "compression_ratio": round(ratio, 3),
        "per_token_compressed_bytes": round(per_token_bytes, 1),
        "per_token_fp16_bytes": round(per_token_fp16, 1),
        "quantize_time_ms": round(quant_time * 1000, 2),
        "k_bits": k_bits,
        "v_bits": v_bits,
    }


def measure_cache_growth(
    n_kv_heads: int,
    head_dim: int,
    k_bits: int = 4,
    v_bits: int = 4,
):
    """Measure KV cache growth as tokens are added incrementally."""
    config = TurboQuantConfig(head_dim=head_dim, k_bits=k_bits, v_bits=v_bits)
    batch = 1
    results = []

    for seq_len in [100, 500, 1000, 2000, 4000, 8000, 16000]:
        keys = mx.random.normal(shape=(batch, n_kv_heads, seq_len, head_dim)).astype(mx.float16)
        values = mx.random.normal(shape=(batch, n_kv_heads, seq_len, head_dim)).astype(mx.float16)
        mx.eval(keys, values)

        quantized = quantize_kv(keys, values, config)
        mx.eval(*quantized.values())

        comp_bytes = kv_bytes(quantized)
        fp16_bytes_val = fp16_kv_bytes(batch, n_kv_heads, seq_len, head_dim)
        ratio = fp16_bytes_val / comp_bytes

        results.append({
            "tokens": seq_len,
            "compressed_mb": round(comp_bytes / (1024 * 1024), 3),
            "fp16_mb": round(fp16_bytes_val / (1024 * 1024), 3),
            "ratio": round(ratio, 3),
        })

    return results


# Model configs matching harness (from scripts/prepare.py get_model_config)
MODEL_CONFIGS = {
    "S": {"name": "Qwen2.5-0.5B", "n_kv_heads": 2, "head_dim": 64, "n_layers": 24},
    "M": {"name": "Qwen2.5-3B", "n_kv_heads": 2, "head_dim": 128, "n_layers": 36},
    "XL": {"name": "Qwen2.5-14B", "n_kv_heads": 4, "head_dim": 128, "n_layers": 48},
}


def main():
    parser = argparse.ArgumentParser(description="TurboQuant memory benchmark")
    parser.add_argument("--tier", choices=["S", "M", "XL", "all"], default="all")
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 6, 8])
    args = parser.parse_args()

    tiers = list(MODEL_CONFIGS.keys()) if args.tier == "all" else [args.tier]
    env = get_environment_info()

    print("=== TurboQuant Memory Benchmark ===\n")

    for tier in tiers:
        cfg = MODEL_CONFIGS[tier]
        print(f"--- {tier}: {cfg['name']} ({cfg['n_kv_heads']} KV heads, dim={cfg['head_dim']}) ---")

        # Compression at various context lengths
        n_layers = cfg["n_layers"]
        print(f"\n  Compression ratio ({args.bits}-bit), per-layer and total ({n_layers} layers):")
        print(f"  {'Context':>8} {'FP16/lyr':>10} {'Comp/lyr':>10} {'Ratio':>8} {'FP16 total':>12} {'Comp total':>12} {'Quant ms':>10}")
        print(f"  {'-'*72}")

        for seq_len in [512, 2048, 4096, 8192, 16384]:
            result = measure_compression(
                n_kv_heads=cfg["n_kv_heads"],
                head_dim=cfg["head_dim"],
                seq_len=seq_len,
                k_bits=args.bits,
                v_bits=args.bits,
            )
            fp16_mb = result["fp16_bytes"] / (1024 * 1024)
            comp_mb = result["compressed_bytes"] / (1024 * 1024)
            fp16_total = fp16_mb * n_layers
            comp_total = comp_mb * n_layers
            result["fp16_total_mb"] = round(fp16_total, 2)
            result["compressed_total_mb"] = round(comp_total, 2)
            result["n_layers"] = n_layers
            print(f"  {seq_len:>8} {fp16_mb:>10.2f} {comp_mb:>10.2f} {result['compression_ratio']:>8.2f}x {fp16_total:>12.1f} {comp_total:>12.1f} {result['quantize_time_ms']:>10.1f}")

            log_experiment(
                experiment_name=f"turbo_quant_memory_{tier.lower()}_k{args.bits}v{args.bits}_{seq_len}",
                phase="turbo_quant_memory",
                config={
                    "tier": tier,
                    "model": cfg["name"],
                    "n_kv_heads": cfg["n_kv_heads"],
                    "head_dim": cfg["head_dim"],
                    "k_bits": args.bits,
                    "v_bits": args.bits,
                    "seq_len": seq_len,
                },
                results=result,
                env=env,
            )

        # Cache growth tracking
        print(f"\n  Cache growth ({args.bits}-bit):")
        growth = measure_cache_growth(
            n_kv_heads=cfg["n_kv_heads"],
            head_dim=cfg["head_dim"],
            k_bits=args.bits,
            v_bits=args.bits,
        )
        for g in growth:
            print(f"    {g['tokens']:>6} tokens: {g['compressed_mb']:>8.3f} MB compressed, {g['fp16_mb']:>8.3f} MB FP16, {g['ratio']:.2f}x")

        print()

    print("=== Done ===")


if __name__ == "__main__":
    main()
