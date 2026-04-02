"""
Phase 0: Memory Budget Modeling for SSD-offloaded dense inference (H8).

Computes exact per-block sizes for Qwen2.5-72B at each bit-width,
models resident/streaming splits, and finds a viable configuration
that fits within the 24 GB M4 Pro memory budget.

Usage:
    uv run python scripts/ssd_lod_memory_budget.py
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.experiment_utils import get_environment_info, log_experiment

# ---------------------------------------------------------------------------
# Qwen2.5-72B architecture (from config.json)
# ---------------------------------------------------------------------------

CONFIG = {
    "num_hidden_layers": 80,
    "hidden_size": 8192,
    "intermediate_size": 29568,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 128,  # 8192 / 64
    "vocab_size": 152064,
    "tie_word_embeddings": False,
}


def params_per_block(cfg: dict) -> dict:
    """Compute parameter count per tensor in one transformer block."""
    h = cfg["hidden_size"]
    i = cfg["intermediate_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]

    # Attention
    q_proj = h * (n_heads * head_dim)       # 8192 * 8192 = 67,108,864
    k_proj = h * (n_kv_heads * head_dim)    # 8192 * 1024 = 8,388,608
    v_proj = h * (n_kv_heads * head_dim)    # 8192 * 1024 = 8,388,608
    o_proj = (n_heads * head_dim) * h       # 8192 * 8192 = 67,108,864

    # Biases (Qwen2 has biases on q/k/v projections)
    q_bias = n_heads * head_dim             # 8192
    k_bias = n_kv_heads * head_dim          # 1024
    v_bias = n_kv_heads * head_dim          # 1024

    # MLP (SwiGLU: gate_proj, up_proj, down_proj)
    gate_proj = h * i                        # 8192 * 29568 = 242,221,056
    up_proj = h * i                          # 8192 * 29568 = 242,221,056
    down_proj = i * h                        # 29568 * 8192 = 242,221,056

    # LayerNorm (tiny — 2 × hidden_size)
    layernorm = 2 * h                        # 16,384

    return {
        "q_proj": q_proj,
        "k_proj": k_proj,
        "v_proj": v_proj,
        "o_proj": o_proj,
        "q_bias": q_bias,
        "k_bias": k_bias,
        "v_bias": v_bias,
        "gate_proj": gate_proj,
        "up_proj": up_proj,
        "down_proj": down_proj,
        "layernorm": layernorm,
        "attention_weights": q_proj + k_proj + v_proj + o_proj,
        "attention_biases": q_bias + k_bias + v_bias,
        "mlp_weights": gate_proj + up_proj + down_proj,
        "total_weights": q_proj + k_proj + v_proj + o_proj + gate_proj + up_proj + down_proj,
        "total_biases": q_bias + k_bias + v_bias,
        "total_params": q_proj + k_proj + v_proj + o_proj + gate_proj + up_proj + down_proj + q_bias + k_bias + v_bias + layernorm,
    }


def bytes_at_bpw(n_params: int, bpw: float, group_size: int = 64) -> float:
    """Compute bytes for n_params at given bits-per-weight, including quant metadata.

    For grouped quantization:
    - Weight data: n_params * bpw / 8
    - Group scales: (n_params / group_size) * 2 bytes (float16)
    - Group zeros: (n_params / group_size) * 2 bytes (float16) [for asymmetric]
    """
    weight_bytes = n_params * bpw / 8
    n_groups = math.ceil(n_params / group_size)
    # MLX quantization: scales (float16) + biases (float16) per group
    scale_bytes = n_groups * 2
    bias_bytes = n_groups * 2
    return weight_bytes + scale_bytes + bias_bytes


def bytes_at_bpw_no_quant(n_params: int, bpw: float) -> float:
    """Bytes for biases/layernorm (stored at full precision, no quantization metadata)."""
    return n_params * bpw / 8


def compute_block_sizes(cfg: dict) -> dict:
    """Compute per-block sizes at various bit-widths."""
    block = params_per_block(cfg)
    results = {}

    for bpw in [2, 3, 4, 6, 8]:
        # Weights are quantized
        weight_bytes = bytes_at_bpw(block["total_weights"], bpw)
        # Biases stay at FP16 (too small to quantize)
        bias_bytes = bytes_at_bpw_no_quant(block["total_biases"], 16)
        # LayerNorm stays at FP16
        ln_bytes = bytes_at_bpw_no_quant(block["layernorm"], 16)

        total = weight_bytes + bias_bytes + ln_bytes
        results[f"Q{bpw}"] = {
            "weight_bytes": weight_bytes,
            "bias_bytes": bias_bytes,
            "layernorm_bytes": ln_bytes,
            "total_bytes": total,
            "total_mb": total / (1024 * 1024),
        }

    return results


def compute_fixed_costs(cfg: dict) -> dict:
    """Compute non-layer costs: embeddings, lm_head, final layernorm."""
    vocab = cfg["vocab_size"]
    h = cfg["hidden_size"]

    # Embeddings at Q6 (common for MLX)
    embed_params = vocab * h  # 152064 * 8192 = 1,245,708,288
    embed_bytes = bytes_at_bpw(embed_params, 6)

    # LM head at Q6 (separate from embeddings since tie_word_embeddings=False)
    lm_head_params = vocab * h
    lm_head_bytes = bytes_at_bpw(lm_head_params, 6)

    # Final layernorm at FP16
    final_ln_bytes = h * 2  # 16,384 bytes

    return {
        "embed_params": embed_params,
        "embed_bytes": embed_bytes,
        "embed_mb": embed_bytes / (1024 * 1024),
        "lm_head_params": lm_head_params,
        "lm_head_bytes": lm_head_bytes,
        "lm_head_mb": lm_head_bytes / (1024 * 1024),
        "final_ln_bytes": final_ln_bytes,
        "total_fixed_bytes": embed_bytes + lm_head_bytes + final_ln_bytes,
        "total_fixed_mb": (embed_bytes + lm_head_bytes + final_ln_bytes) / (1024 * 1024),
    }


def compute_kv_cache(cfg: dict, context_len: int, kv_bits: int = 16) -> dict:
    """Compute KV cache size."""
    n_layers = cfg["num_hidden_layers"]
    n_kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]

    # KV cache: 2 (K+V) × n_layers × n_kv_heads × head_dim × context_len × bits/8
    kv_bytes = 2 * n_layers * n_kv_heads * head_dim * context_len * kv_bits / 8
    return {
        "kv_bytes": kv_bytes,
        "kv_mb": kv_bytes / (1024 * 1024),
        "kv_gb": kv_bytes / (1024 * 1024 * 1024),
        "context_len": context_len,
        "kv_bits": kv_bits,
    }


def evaluate_split(
    cfg: dict,
    n_resident_first: int,
    n_resident_last: int,
    resident_bpw: int,
    streaming_bpw: int,
    context_len: int = 2048,
    kv_bits: int = 16,
    runtime_overhead_gb: float = 2.0,
    os_overhead_gb: float = 3.0,
    total_ram_gb: float = 24.0,
) -> dict:
    """Evaluate a specific resident/streaming split configuration."""
    n_layers = cfg["num_hidden_layers"]
    n_streaming = n_layers - n_resident_first - n_resident_last

    block_sizes = compute_block_sizes(cfg)
    fixed = compute_fixed_costs(cfg)
    kv = compute_kv_cache(cfg, context_len, kv_bits)

    resident_block_bytes = block_sizes[f"Q{resident_bpw}"]["total_bytes"]
    streaming_block_bytes = block_sizes[f"Q{streaming_bpw}"]["total_bytes"]

    total_resident_blocks = (n_resident_first + n_resident_last) * resident_block_bytes
    total_streaming_blocks = n_streaming * streaming_block_bytes

    # Pinned memory
    pinned_weights = total_resident_blocks + fixed["total_fixed_bytes"]
    pinned_kv = kv["kv_bytes"]
    pinned_runtime = runtime_overhead_gb * 1024 * 1024 * 1024  # Metal + Python/MLX
    pinned_total = pinned_weights + pinned_kv + pinned_runtime
    pinned_total_gb = pinned_total / (1024**3)

    os_bytes = os_overhead_gb * 1024**3
    total_pinned_plus_os_gb = (pinned_total + os_bytes) / (1024**3)

    available_for_cache_gb = total_ram_gb - total_pinned_plus_os_gb
    closes = total_pinned_plus_os_gb <= (total_ram_gb - 1.0)  # Must leave ≥1 GB headroom

    # Average bpw across entire model
    total_weight_params = params_per_block(cfg)["total_weights"] * n_layers
    resident_bits = (n_resident_first + n_resident_last) * params_per_block(cfg)["total_weights"] * resident_bpw
    streaming_bits = n_streaming * params_per_block(cfg)["total_weights"] * streaming_bpw
    avg_bpw = (resident_bits + streaming_bits) / total_weight_params

    label = f"{n_resident_first}+{n_resident_last} Q{resident_bpw} / {n_streaming} Q{streaming_bpw}"

    return {
        "label": label,
        "n_resident_first": n_resident_first,
        "n_resident_last": n_resident_last,
        "n_streaming": n_streaming,
        "resident_bpw": resident_bpw,
        "streaming_bpw": streaming_bpw,
        "avg_bpw": round(avg_bpw, 2),
        "context_len": context_len,
        "kv_bits": kv_bits,
        "resident_blocks_gb": round(total_resident_blocks / (1024**3), 2),
        "fixed_costs_gb": round(fixed["total_fixed_bytes"] / (1024**3), 2),
        "kv_cache_gb": round(kv["kv_gb"], 2),
        "runtime_overhead_gb": runtime_overhead_gb,
        "pinned_total_gb": round(pinned_total_gb, 2),
        "os_overhead_gb": os_overhead_gb,
        "total_pinned_plus_os_gb": round(total_pinned_plus_os_gb, 2),
        "available_for_cache_gb": round(available_for_cache_gb, 2),
        "streaming_blocks_gb": round(total_streaming_blocks / (1024**3), 2),
        "total_model_gb": round((total_resident_blocks + total_streaming_blocks + fixed["total_fixed_bytes"]) / (1024**3), 2),
        "closes": closes,
        "block_size_resident_mb": round(resident_block_bytes / (1024**2), 1),
        "block_size_streaming_mb": round(streaming_block_bytes / (1024**2), 1),
    }


def main():
    print("=" * 70)
    print("Phase 0: Memory Budget Modeling for SSD-Offloaded Dense Inference")
    print("=" * 70)

    env = get_environment_info()
    print(f"Environment: {env['chip']}, {env['memory_gb']} GB RAM, {env['available_gb']} GB available")

    cfg = CONFIG
    n_layers = cfg["num_hidden_layers"]

    # --- Per-block sizes ---
    print(f"\n{'='*70}")
    print(f"Per-Block Sizes (Qwen2.5-72B, {n_layers} blocks)")
    print(f"{'='*70}")

    block = params_per_block(cfg)
    print(f"\nParams per block:")
    print(f"  Attention weights: {block['attention_weights']:>15,} ({block['attention_weights']/1e9:.2f}B)")
    print(f"  Attention biases:  {block['attention_biases']:>15,}")
    print(f"  MLP weights:       {block['mlp_weights']:>15,} ({block['mlp_weights']/1e9:.2f}B)")
    print(f"  LayerNorm:         {block['layernorm']:>15,}")
    print(f"  Total params:      {block['total_params']:>15,} ({block['total_params']/1e9:.2f}B)")
    print(f"  Total weights:     {block['total_weights']:>15,} ({block['total_weights']/1e9:.2f}B)")

    block_sizes = compute_block_sizes(cfg)
    print(f"\nBlock size at each bit-width:")
    for bpw_label, info in block_sizes.items():
        print(f"  {bpw_label}: {info['total_mb']:>8.1f} MB  ({info['total_bytes']:>12,} bytes)")

    # --- Fixed costs ---
    fixed = compute_fixed_costs(cfg)
    print(f"\nFixed costs:")
    print(f"  Embeddings (Q6):  {fixed['embed_mb']:>8.1f} MB  ({fixed['embed_params']:,} params)")
    print(f"  LM Head (Q6):     {fixed['lm_head_mb']:>8.1f} MB  ({fixed['lm_head_params']:,} params)")
    print(f"  Total fixed:      {fixed['total_fixed_mb']:>8.1f} MB")

    # --- KV cache sizes ---
    print(f"\nKV cache sizes:")
    for ctx_len in [512, 1024, 2048, 4096]:
        for kv_bits in [4, 8, 16]:
            kv = compute_kv_cache(cfg, ctx_len, kv_bits)
            print(f"  ctx={ctx_len:>5}, Q{kv_bits:>2}: {kv['kv_mb']:>8.1f} MB ({kv['kv_gb']:.2f} GB)")

    # --- Evaluate splits ---
    print(f"\n{'='*70}")
    print(f"Split Configurations")
    print(f"{'='*70}")

    splits = [
        # Original plan: 20/60/20 at Q4/Q2/Q4
        (16, 16, 4, 2, 2048, 16),
        # Mitigation: 10/80/10 at Q4/Q2/Q4
        (8, 8, 4, 2, 2048, 16),
        # Mitigation: 10/80/10 at Q3/Q2/Q3
        (8, 8, 3, 2, 2048, 16),
        # Mitigation: 5/90/5 at Q4/Q2/Q4
        (4, 4, 4, 2, 2048, 16),
        # Mitigation: 5/90/5 at Q3/Q2/Q3
        (4, 4, 3, 2, 2048, 16),
        # With KV cache compression
        (8, 8, 4, 2, 2048, 4),
        (8, 8, 3, 2, 2048, 4),
        (4, 4, 4, 2, 2048, 4),
        # Shorter context
        (8, 8, 4, 2, 512, 16),
        (8, 8, 3, 2, 512, 16),
        # Aggressive: all Q2 streamed except last 3 blocks
        (0, 3, 4, 2, 2048, 16),
        (0, 3, 4, 2, 2048, 4),
        # Middle ground: protect only last 5 + first 3
        (3, 5, 4, 2, 2048, 16),
        (3, 5, 4, 2, 2048, 4),
        (3, 5, 3, 2, 2048, 4),
        # Uniform Q2 (no streaming needed — fits?)
        (80, 0, 2, 2, 2048, 16),
        (80, 0, 2, 2, 2048, 4),
    ]

    results = []
    viable = []

    print(f"\n{'Label':<40} {'Avg BPW':>8} {'Pinned':>8} {'Pin+OS':>8} {'Cache':>8} {'Stream':>8} {'Closes':>7}")
    print("-" * 100)

    for n_first, n_last, res_bpw, str_bpw, ctx, kv_bits in splits:
        r = evaluate_split(cfg, n_first, n_last, res_bpw, str_bpw, ctx, kv_bits)
        results.append(r)

        status = "YES" if r["closes"] else "NO"
        flag = " ***" if r["closes"] else ""
        print(f"  {r['label']:<38} {r['avg_bpw']:>7.2f} {r['pinned_total_gb']:>7.1f}G {r['total_pinned_plus_os_gb']:>7.1f}G {r['available_for_cache_gb']:>7.1f}G {r['streaming_blocks_gb']:>7.1f}G {status:>6}{flag}")

        if r["closes"]:
            viable.append(r)

    # --- Summary ---
    print(f"\n{'='*70}")
    print(f"VIABLE CONFIGURATIONS ({len(viable)} found)")
    print(f"{'='*70}")

    if not viable:
        print("\nNO VIABLE CONFIGURATION FOUND.")
        print("The experiment cannot proceed to Phase 3 without additional mitigations.")
        print("Consider: external memory (swap), reduced model size (32B), or accepting OOM risk.")
    else:
        for r in viable:
            print(f"\n  {r['label']}")
            print(f"    Average BPW:           {r['avg_bpw']}")
            print(f"    Resident blocks:       {r['n_resident_first']}+{r['n_resident_last']} = {r['n_resident_first']+r['n_resident_last']} blocks ({r['resident_blocks_gb']} GB)")
            print(f"    Fixed costs:           {r['fixed_costs_gb']} GB (embeddings + lm_head)")
            print(f"    KV cache:              {r['kv_cache_gb']} GB (ctx={r['context_len']}, Q{r['kv_bits']})")
            print(f"    Runtime overhead:       {r['runtime_overhead_gb']} GB")
            print(f"    Total pinned:          {r['pinned_total_gb']} GB")
            print(f"    Total pinned + OS:     {r['total_pinned_plus_os_gb']} GB")
            print(f"    Available for cache:   {r['available_for_cache_gb']} GB")
            print(f"    Streaming on SSD:      {r['streaming_blocks_gb']} GB ({r['n_streaming']} blocks)")
            print(f"    Total model size:      {r['total_model_gb']} GB")
            print(f"    Block size (resident): {r['block_size_resident_mb']} MB")
            print(f"    Block size (stream):   {r['block_size_streaming_mb']} MB")

        # Pick the best viable config: highest avg_bpw that closes
        best = max(viable, key=lambda r: (r["avg_bpw"], r["available_for_cache_gb"]))
        print(f"\n  RECOMMENDED: {best['label']}")
        print(f"    This gives the highest quality (avg {best['avg_bpw']} bpw) while closing the budget.")

    # --- 16 GB analysis ---
    print(f"\n{'='*70}")
    print(f"16 GB FEASIBILITY (exploratory)")
    print(f"{'='*70}")

    splits_16gb = [
        (0, 3, 4, 2, 512, 4),
        (0, 3, 3, 2, 512, 4),
        (0, 0, 2, 2, 512, 4),  # uniform Q2, no resident, all streamed
    ]

    print(f"\n{'Label':<40} {'Pinned':>8} {'Pin+OS':>8} {'Closes (16GB)':>14}")
    print("-" * 75)

    for n_first, n_last, res_bpw, str_bpw, ctx, kv_bits in splits_16gb:
        r = evaluate_split(cfg, n_first, n_last, res_bpw, str_bpw, ctx, kv_bits,
                          total_ram_gb=16.0, os_overhead_gb=3.0)
        status = "YES" if r["closes"] else f"NO (gap: {r['total_pinned_plus_os_gb'] - 15.0:.1f}G)"
        print(f"  {r['label']:<38} {r['pinned_total_gb']:>7.1f}G {r['total_pinned_plus_os_gb']:>7.1f}G {status:>13}")

    # --- Log results ---
    log_experiment(
        experiment_name="ssd_lod_memory_budget_72b",
        phase="memory_budget",
        config={
            "model": "Qwen2.5-72B-Instruct",
            "architecture": cfg,
        },
        results={
            "block_params": params_per_block(cfg),
            "block_sizes": {k: v["total_mb"] for k, v in block_sizes.items()},
            "fixed_costs_mb": fixed["total_fixed_mb"],
            "n_viable_configs": len(viable),
            "viable_configs": viable,
            "recommended": best if viable else None,
            "pass_fail": "PASS" if viable else "FAIL",
        },
    )

    print(f"\nResults logged to experiments.jsonl")

    # --- Output Phase 0 parameters for downstream phases ---
    if viable:
        print(f"\n{'='*70}")
        print(f"PHASE 0 OUTPUT (for downstream phases)")
        print(f"{'='*70}")
        print(f"  TARGET_AVAILABLE_GB={best['available_for_cache_gb']}")
        print(f"  N_STREAMING_BLOCKS={best['n_streaming']}")
        print(f"  BLOCK_SIZE_MB={best['block_size_streaming_mb']}")
        print(f"  PINNED_TOTAL_GB={best['pinned_total_gb']}")
        print(f"  RESIDENT_SPLIT={best['n_resident_first']}+{best['n_resident_last']}")
        print(f"  RESIDENT_BPW={best['resident_bpw']}")
        print(f"  STREAMING_BPW={best['streaming_bpw']}")
        print(f"  CONTEXT_LEN={best['context_len']}")
        print(f"  KV_BITS={best['kv_bits']}")


if __name__ == "__main__":
    main()
