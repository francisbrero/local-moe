"""
Phase 2-3: Layer LOD Allocation Comparison + Benchmark.

Compares multiple quantization allocation strategies on the same model:
1. Uniform Q4 (baseline)
2. Uniform Q2 (lower bound)
3. U-shape: Q4 for first/last 20%, Q2 for middle 60%
4. Sensitivity-guided: greedy knapsack from Phase 1 data
5. mlx-lm built-in mixed_2_6 recipe

All comparisons are at matched memory budgets where applicable.

Usage:
    uv run python scripts/layer_lod_bench.py [--model MODEL_ID]
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.experiment_utils import get_environment_info, log_experiment
from scripts.layer_sensitivity import (
    compute_perplexity,
    get_quantized_layer_names,
    get_layer_index,
    requantize_layer,
    restore_layer,
    load_calibration_data,
    _navigate_to_module,
)


# ---------------------------------------------------------------------------
# Allocation strategies
# ---------------------------------------------------------------------------


def make_ushape_allocation(n_blocks: int, edge_pct: float = 0.2) -> dict[int, int]:
    """U-shape: Q4 for first/last edge_pct, Q2 for middle."""
    n_edge = max(1, int(n_blocks * edge_pct))
    alloc = {}
    for i in range(n_blocks):
        if i < n_edge or i >= n_blocks - n_edge:
            alloc[i] = 4
        else:
            alloc[i] = 2
    return alloc


def make_gradient_allocation(n_blocks: int) -> dict[int, int]:
    """Gradient: Q4 → Q3 → Q2 → Q3 → Q4 (smooth ramp)."""
    alloc = {}
    mid = n_blocks / 2
    for i in range(n_blocks):
        dist_from_edge = min(i, n_blocks - 1 - i)
        # Map distance to bits: 0 → 4, mid → 2
        ratio = dist_from_edge / max(mid, 1)
        bits = max(2, min(4, round(4 - 2 * ratio)))
        alloc[i] = bits
    return alloc


def make_sensitivity_allocation(
    sensitivity_data: list[dict],
    target_avg_bpw: float = 3.0,
) -> dict[int, int]:
    """Greedy knapsack: assign bits based on sensitivity data from Phase 1.

    Uses Q2 sensitivity deltas to rank blocks (largest positive delta = most
    sensitive). Upgrades from Q2 to Q4 starting with the most sensitive until
    the bit budget is exhausted. Blocks with negative deltas (those that
    actually improve at Q2) naturally stay at Q2 since they sort last.
    """
    # Sort by sensitivity (highest delta first)
    sorted_blocks = sorted(sensitivity_data, key=lambda x: x["ppl_delta"], reverse=True)
    n_blocks = len(sorted_blocks)

    # Start all blocks at Q2 (2 bpw)
    alloc = {b["block_index"]: 2 for b in sorted_blocks}

    # Bit budget: (target_avg - 2) * n_blocks bits to distribute
    budget = (target_avg_bpw - 2.0) * n_blocks

    # Greedily upgrade most sensitive blocks to Q4 (cost = 2 bits each)
    for block in sorted_blocks:
        if budget < 2:
            break
        alloc[block["block_index"]] = 4
        budget -= 2

    # If budget remains, upgrade some to Q3 (cost = 1 bit from Q2)
    if budget > 0:
        for block in sorted_blocks:
            if budget < 1:
                break
            if alloc[block["block_index"]] == 2:
                alloc[block["block_index"]] = 3
                budget -= 1

    return alloc


def apply_allocation(model, block_layers: dict[int, list[str]], allocation: dict[int, int]):
    """Apply a block-level allocation, re-quantizing each block to its assigned bits.

    Returns dict of {layer_name: original_state} for restoration.
    """
    originals = {}
    for block_idx, target_bits in allocation.items():
        if block_idx not in block_layers:
            continue
        for layer_name in block_layers[block_idx]:
            module = _navigate_to_module(model, layer_name)
            if not isinstance(module, nn.QuantizedLinear):
                continue
            # Only re-quantize if different from current
            if module.bits != target_bits:
                orig = requantize_layer(model, layer_name, target_bits)
                if orig:
                    originals[layer_name] = orig
    return originals


def compute_allocation_bpw(allocation: dict[int, int]) -> float:
    """Compute average bits-per-weight for an allocation."""
    if not allocation:
        return 0
    return sum(allocation.values()) / len(allocation)


def compute_allocation_bytes(allocation: dict[int, int], params_per_block: int) -> int:
    """Compute total trunk bytes for an allocation."""
    total_bits = sum(bits * params_per_block for bits in allocation.values())
    return total_bits // 8


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def benchmark_allocation(
    model,
    tokenizer,
    texts: list[str],
    name: str,
    block_layers: dict[int, list[str]],
    allocation: dict[int, int],
    max_len: int = 512,
) -> dict:
    """Benchmark a single allocation strategy."""
    from mlx_lm import generate

    print(f"\n  [{name}] avg bpw = {compute_allocation_bpw(allocation):.2f}")

    # Apply allocation
    originals = apply_allocation(model, block_layers, allocation)

    # Measure perplexity
    t0 = time.time()
    ppl = compute_perplexity(model, tokenizer, texts, max_len)
    ppl_time = time.time() - t0

    # Measure throughput (generate 50 tokens)
    prompt = "Explain the concept of quantization in neural networks."
    tokens_in = tokenizer.encode(prompt)

    # Warm up
    generate(model, tokenizer, prompt=prompt, max_tokens=10, verbose=False)

    # Measure
    mx.synchronize()
    t0 = time.time()
    output = generate(model, tokenizer, prompt=prompt, max_tokens=50, verbose=False)
    mx.synchronize()
    gen_time = time.time() - t0
    tok_s = 50 / gen_time if gen_time > 0 else 0

    # Memory
    rss_mb = psutil.Process().memory_info().rss / (1024**2)
    available_gb = psutil.virtual_memory().available / (1024**3)

    result = {
        "name": name,
        "avg_bpw": round(compute_allocation_bpw(allocation), 2),
        "perplexity": round(ppl, 4),
        "ppl_eval_time_s": round(ppl_time, 1),
        "tok_s": round(tok_s, 1),
        "gen_time_s": round(gen_time, 1),
        "rss_mb": round(rss_mb, 0),
        "available_gb": round(available_gb, 2),
        "allocation": {str(k): v for k, v in allocation.items()},
    }

    print(f"    PPL: {ppl:.4f}, tok/s: {tok_s:.1f}, "
          f"RSS: {rss_mb:.0f} MB, avail: {available_gb:.1f} GB")

    # Restore original weights
    for layer_name, orig in originals.items():
        restore_layer(model, layer_name, orig)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_phase1_sensitivity() -> list[dict] | None:
    """Load Phase 1 sensitivity data from experiments.jsonl."""
    jsonl_path = Path(__file__).parent.parent / "experiments.jsonl"
    if not jsonl_path.exists():
        return None

    q2_results = None
    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line)
            if (record.get("phase") == "layer_sensitivity" and
                record.get("config", {}).get("target_bits") == 2):
                q2_results = record.get("results", {}).get("block_results", [])

    return q2_results


def main():
    parser = argparse.ArgumentParser(description="Layer LOD Phase 2-3: Allocation Benchmark")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit",
                        help="Model ID to benchmark")
    parser.add_argument("--n-samples", type=int, default=8,
                        help="Number of calibration samples")
    parser.add_argument("--max-len", type=int, default=512,
                        help="Maximum sequence length")
    args = parser.parse_args()

    print("=" * 60)
    print("Layer LOD — Phase 2-3: Allocation Benchmark")
    print("=" * 60)

    env = get_environment_info()
    print(f"Environment: {env['chip']}, {env['memory_gb']} GB RAM")

    # Load Phase 1 sensitivity data
    sensitivity_data = load_phase1_sensitivity()
    if sensitivity_data:
        print(f"\nLoaded Phase 1 sensitivity data: {len(sensitivity_data)} blocks")
    else:
        print("\nNo Phase 1 data found — will skip sensitivity-guided allocation")

    # Load model
    from mlx_lm import load
    print(f"\nLoading model: {args.model}")
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    # Get layer structure
    quant_layers = get_quantized_layer_names(model)
    block_layers = {}
    for name in quant_layers:
        idx = get_layer_index(name)
        if idx >= 0:
            block_layers.setdefault(idx, []).append(name)
    n_blocks = len(block_layers)
    print(f"Transformer blocks: {n_blocks}")

    # Calibration data
    texts = load_calibration_data(tokenizer, args.n_samples)

    # Define allocations to compare
    strategies = {}

    # 1. Uniform Q4 (baseline)
    strategies["uniform_q4"] = {i: 4 for i in range(n_blocks)}

    # 2. Uniform Q2 (lower bound)
    strategies["uniform_q2"] = {i: 2 for i in range(n_blocks)}

    # 3. Uniform Q3 (middle ground)
    strategies["uniform_q3"] = {i: 3 for i in range(n_blocks)}

    # 4. U-shape (Q4 edges, Q2 middle)
    strategies["ushape_20pct"] = make_ushape_allocation(n_blocks, edge_pct=0.2)

    # 5. Gradient (Q4 → Q3 → Q2 → Q3 → Q4)
    strategies["gradient"] = make_gradient_allocation(n_blocks)

    # 6. Sensitivity-guided (if Phase 1 data available)
    if sensitivity_data:
        strategies["sensitivity_3bpw"] = make_sensitivity_allocation(
            sensitivity_data, target_avg_bpw=3.0
        )
        strategies["sensitivity_2.5bpw"] = make_sensitivity_allocation(
            sensitivity_data, target_avg_bpw=2.5
        )

    # Print allocation summaries
    print("\n" + "-" * 60)
    print("ALLOCATION STRATEGIES")
    print("-" * 60)
    for name, alloc in strategies.items():
        bpw = compute_allocation_bpw(alloc)
        bit_counts = {}
        for bits in alloc.values():
            bit_counts[bits] = bit_counts.get(bits, 0) + 1
        summary = ", ".join(f"{count}×Q{bits}" for bits, count in sorted(bit_counts.items()))
        print(f"  {name:25s}: avg {bpw:.2f} bpw — {summary}")

    # Run benchmarks
    print("\n" + "=" * 60)
    print("BENCHMARKS")
    print("=" * 60)

    all_results = []
    for name, alloc in strategies.items():
        result = benchmark_allocation(
            model, tokenizer, texts, name, block_layers, alloc,
            max_len=args.max_len,
        )
        all_results.append(result)

    # Log results
    log_experiment(
        experiment_name=f"layer_lod_bench_{args.model.split('/')[-1]}",
        phase="lod_benchmark",
        config={
            "model_id": args.model,
            "n_samples": args.n_samples,
            "max_len": args.max_len,
            "n_blocks": n_blocks,
            "strategies": list(strategies.keys()),
        },
        results={
            "benchmarks": all_results,
        },
    )

    # Summary table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Strategy':<25s} {'BPW':>5s} {'PPL':>8s} {'Δ PPL':>8s} {'tok/s':>7s}")
    print("-" * 60)

    baseline_ppl = None
    for r in all_results:
        if r["name"] == "uniform_q4":
            baseline_ppl = r["perplexity"]

    for r in all_results:
        delta = r["perplexity"] - baseline_ppl if baseline_ppl else 0
        print(f"{r['name']:<25s} {r['avg_bpw']:>5.2f} {r['perplexity']:>8.4f} "
              f"{delta:>+8.4f} {r['tok_s']:>7.1f}")

    # Pareto analysis: best PPL at each BPW level
    print("\n" + "-" * 60)
    print("PARETO ANALYSIS (best PPL at each BPW)")
    print("-" * 60)

    # Group by approximate BPW
    bpw_groups = {}
    for r in all_results:
        bpw_key = round(r["avg_bpw"] * 2) / 2  # Round to nearest 0.5
        if bpw_key not in bpw_groups or r["perplexity"] < bpw_groups[bpw_key]["perplexity"]:
            bpw_groups[bpw_key] = r

    for bpw in sorted(bpw_groups.keys()):
        r = bpw_groups[bpw]
        print(f"  ~{bpw:.1f} bpw: {r['name']} — PPL {r['perplexity']:.4f}")

    # Key finding: LOD vs uniform at same BPW
    print("\n" + "-" * 60)
    print("KEY FINDING: LOD vs Uniform at Same BPW")
    print("-" * 60)

    for r in all_results:
        if r["name"].startswith("uniform_"):
            continue
        # Find uniform with closest BPW
        closest_uniform = min(
            [u for u in all_results if u["name"].startswith("uniform_")],
            key=lambda u: abs(u["avg_bpw"] - r["avg_bpw"]),
        )
        ppl_improvement = ((closest_uniform["perplexity"] - r["perplexity"])
                          / closest_uniform["perplexity"] * 100)
        print(f"  {r['name']:<25s} ({r['avg_bpw']:.2f} bpw): "
              f"PPL {r['perplexity']:.4f} vs uniform ({closest_uniform['name']}) "
              f"PPL {closest_uniform['perplexity']:.4f} → "
              f"{ppl_improvement:+.1f}% {'✓ BETTER' if ppl_improvement > 0 else '✗ WORSE'}")

    # Clean up
    del model, tokenizer

    print(f"\nDone. Results logged to experiments.jsonl")


if __name__ == "__main__":
    main()
