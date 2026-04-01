"""
Phase 2: Scheduling Strategy Selection for SSD Layer LOD.

Compares serial vs double-buffer (prefetch) streaming strategies.
Uses the 7B model with custom forward pass that loads blocks on-demand.

Strategies:
    serial       — Load block N, compute N, then load N+1 (baseline)
    double-buf   — Background-load block N+1 while computing N

Usage:
    uv run python scripts/ssd_lod_scheduler.py [--strategy STRATEGY] [--n-tokens N] [--pressure-gb GB]
"""

import argparse
import gc
import os
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.experiment_utils import (
    get_environment_info,
    get_rss_mb,
    get_peak_rss_mb,
    get_vm_stat,
    vm_stat_delta,
    log_experiment,
    create_memory_pressure,
    get_available_memory_gb,
)


def get_model_blocks(model):
    """Get the list of transformer block modules."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    return list(model.layers)


def get_inner_model(model):
    """Get the inner model from the causal LM wrapper."""
    if hasattr(model, "model"):
        return model.model
    return model


def save_blocks_to_disk(blocks, block_indices, save_dir: Path):
    """Save specified blocks' weights to disk as numpy files."""
    save_dir.mkdir(parents=True, exist_ok=True)
    for idx in block_indices:
        block_file = save_dir / f"block_{idx:03d}.npz"
        if block_file.exists():
            continue
        flat = {}
        for name, module in blocks[idx].named_modules():
            if isinstance(module, nn.QuantizedLinear):
                flat[f"{name}.weight"] = np.array(module.weight)
                flat[f"{name}.scales"] = np.array(module.scales)
                if hasattr(module, "biases") and module.biases is not None:
                    flat[f"{name}.biases"] = np.array(module.biases)
        np.savez(block_file, **flat)


def load_block_weights_from_disk(block_idx: int, save_dir: Path) -> dict:
    """Load block weights from disk into numpy dict (CPU work, no MLX)."""
    block_file = save_dir / f"block_{block_idx:03d}.npz"
    return dict(np.load(block_file))


def swap_block_weights(block, weights_dict: dict):
    """Swap block weights from numpy dict into MLX tensors."""
    for name, module in block.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            w_key = f"{name}.weight"
            s_key = f"{name}.scales"
            b_key = f"{name}.biases"
            if w_key in weights_dict:
                module.weight = mx.array(weights_dict[w_key])
                module.scales = mx.array(weights_dict[s_key])
                if b_key in weights_dict:
                    module.biases = mx.array(weights_dict[b_key])
    mx.eval(block.parameters())


def evict_block(block):
    """Replace block weights with tiny placeholders."""
    for name, module in block.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            module.weight = mx.zeros((1,), dtype=mx.uint32)
            module.scales = mx.zeros((1,), dtype=mx.float16)
            if hasattr(module, "biases") and module.biases is not None:
                module.biases = mx.zeros((1,), dtype=mx.float16)
    mx.eval(block.parameters())


def serial_forward(model, input_ids, blocks, streaming_indices, save_dir, metrics):
    """Serial strategy: load → compute → evict for each streaming block."""
    inner = get_inner_model(model)
    h = inner.embed_tokens(input_ids)
    mask = None if h.shape[1] == 1 else "causal"
    streaming_set = set(streaming_indices)

    for i, layer in enumerate(inner.layers):
        if i in streaming_set:
            t0 = time.perf_counter()
            weights = load_block_weights_from_disk(i, save_dir)
            swap_block_weights(blocks[i], weights)
            load_ms = (time.perf_counter() - t0) * 1000
            metrics["load_ms"].append(load_ms)

        h = layer(h, mask, None)

        if i in streaming_set:
            mx.eval(h)
            evict_block(blocks[i])

    h = inner.norm(h)
    if hasattr(model, "args") and model.args.tie_word_embeddings:
        return inner.embed_tokens.as_linear(h)
    return model.lm_head(h)


def double_buffer_forward(model, input_ids, blocks, streaming_indices, save_dir, metrics, executor):
    """Double-buffer strategy: prefetch next block while computing current.

    Uses a background thread to load block N+1's weights from disk
    while the GPU processes block N. The executor should be created once
    outside the token loop and reused across calls.
    """
    inner = get_inner_model(model)
    h = inner.embed_tokens(input_ids)
    mask = None if h.shape[1] == 1 else "causal"
    streaming_set = set(streaming_indices)

    # Find ordered list of streaming block indices
    stream_order = sorted(streaming_indices)
    # Map block index to next streaming block index
    next_stream = {}
    for j in range(len(stream_order) - 1):
        next_stream[stream_order[j]] = stream_order[j + 1]

    # Prefetch state
    pending_future = None
    pending_idx = None

    # Kick off prefetch for the very first streaming block
    if stream_order:
        first_idx = stream_order[0]
        pending_future = executor.submit(load_block_weights_from_disk, first_idx, save_dir)
        pending_idx = first_idx

    for i, layer in enumerate(inner.layers):
        if i in streaming_set:
            # Wait for this block's prefetch to complete
            if pending_idx == i and pending_future is not None:
                t_wait = time.perf_counter()
                weights = pending_future.result()
                wait_ms = (time.perf_counter() - t_wait) * 1000
                metrics["wait_ms"].append(wait_ms)
                pending_future = None
                pending_idx = None
            else:
                # Fallback: no prefetch available, load synchronously
                t0 = time.perf_counter()
                weights = load_block_weights_from_disk(i, save_dir)
                load_ms = (time.perf_counter() - t0) * 1000
                metrics["wait_ms"].append(load_ms)

            t_swap = time.perf_counter()
            swap_block_weights(blocks[i], weights)
            swap_ms = (time.perf_counter() - t_swap) * 1000
            metrics["load_ms"].append(swap_ms)

        # Start computing this layer
        h = layer(h, mask, None)

        if i in streaming_set:
            # While GPU computes, prefetch the next streaming block
            if i in next_stream:
                next_idx = next_stream[i]
                pending_future = executor.submit(load_block_weights_from_disk, next_idx, save_dir)
                pending_idx = next_idx

            mx.eval(h)
            evict_block(blocks[i])

    h = inner.norm(h)
    if hasattr(model, "args") and model.args.tie_word_embeddings:
        return inner.embed_tokens.as_linear(h)
    return model.lm_head(h)


def run_strategy_benchmark(
    model, tokenizer, blocks, strategy, n_tokens, save_dir,
    resident_first_pct=0.1, resident_last_pct=0.1,
) -> dict:
    """Run a scheduling strategy benchmark."""
    n_blocks = len(blocks)
    n_first = int(n_blocks * resident_first_pct)
    n_last = int(n_blocks * resident_last_pct)
    resident_indices = list(range(n_first)) + list(range(n_blocks - n_last, n_blocks))
    streaming_indices = list(range(n_first, n_blocks - n_last))

    print(f"\n  Strategy: {strategy}")
    print(f"  Resident: {len(resident_indices)} blocks, Streaming: {len(streaming_indices)} blocks")

    # Save and evict streaming blocks
    save_blocks_to_disk(blocks, streaming_indices, save_dir)
    for idx in streaming_indices:
        evict_block(blocks[idx])
    gc.collect()
    mx.clear_cache()

    vm_before = get_vm_stat()
    rss_before = get_rss_mb()
    available_before = get_available_memory_gb()
    print(f"  RSS: {rss_before:.0f} MB, Available: {available_before:.1f} GB")

    prompt = "The transformer architecture has revolutionized natural language processing because"
    tokens = tokenizer.encode(prompt)

    per_token_ms = []
    all_metrics = {"load_ms": [], "wait_ms": []}

    # Create executor once for double-buffer (avoids per-token thread creation overhead)
    executor = ThreadPoolExecutor(max_workers=1) if strategy == "double-buf" else None

    for tok_i in range(n_tokens):
        t0 = time.perf_counter()
        input_ids = mx.array([tokens[-1:]])

        token_metrics = {"load_ms": [], "wait_ms": []}
        if strategy == "serial":
            logits = serial_forward(model, input_ids, blocks, streaming_indices, save_dir, token_metrics)
        else:
            logits = double_buffer_forward(model, input_ids, blocks, streaming_indices, save_dir, token_metrics, executor)

        next_token = mx.argmax(logits[0, -1, :]).item()  # .item() forces eval
        tokens.append(next_token)

        tok_ms = (time.perf_counter() - t0) * 1000
        per_token_ms.append(tok_ms)
        all_metrics["load_ms"].extend(token_metrics["load_ms"])
        all_metrics["wait_ms"].extend(token_metrics["wait_ms"])

        if tok_i < 3 or tok_i % 10 == 0:
            rss = get_rss_mb()
            total_load = sum(token_metrics["load_ms"])
            total_wait = sum(token_metrics["wait_ms"]) if token_metrics["wait_ms"] else 0
            print(f"    Token {tok_i}: {tok_ms:.0f} ms (load={total_load:.0f} ms, wait={total_wait:.0f} ms), RSS={rss:.0f} MB")

    if executor is not None:
        executor.shutdown(wait=True)

    vm_after = get_vm_stat()
    rss_after = get_rss_mb()
    peak_rss = get_peak_rss_mb()

    # Restore blocks
    for idx in streaming_indices:
        weights = load_block_weights_from_disk(idx, save_dir)
        swap_block_weights(blocks[idx], weights)

    # Stats (skip first 2 warmup)
    steady = per_token_ms[2:] if len(per_token_ms) > 2 else per_token_ms
    avg_tok_ms = sum(steady) / len(steady) if steady else 0
    tok_s = 1000 / avg_tok_ms if avg_tok_ms > 0 else 0

    load_times = all_metrics["load_ms"]
    wait_times = all_metrics["wait_ms"]

    vm_delta = vm_stat_delta(vm_before, vm_after)

    result = {
        "strategy": strategy,
        "n_tokens": n_tokens,
        "n_streaming": len(streaming_indices),
        "tok_s": round(tok_s, 2),
        "avg_tok_ms": round(avg_tok_ms, 1),
        "peak_rss_mb": round(peak_rss, 1),
        "load_p50_ms": round(float(np.percentile(load_times, 50)), 1) if load_times else 0,
        "load_p95_ms": round(float(np.percentile(load_times, 95)), 1) if load_times else 0,
        "wait_p50_ms": round(float(np.percentile(wait_times, 50)), 1) if wait_times else 0,
        "wait_p95_ms": round(float(np.percentile(wait_times, 95)), 1) if wait_times else 0,
        "pageout_delta_mb": vm_delta["pageout_delta_mb"],
        "pagein_delta_mb": vm_delta["pagein_delta_mb"],
    }

    print(f"\n  Results ({strategy}):")
    print(f"    tok/s: {tok_s:.2f} ({avg_tok_ms:.0f} ms/tok)")
    print(f"    Load p50/p95: {result['load_p50_ms']:.1f}/{result['load_p95_ms']:.1f} ms")
    if wait_times:
        print(f"    Wait p50/p95: {result['wait_p50_ms']:.1f}/{result['wait_p95_ms']:.1f} ms")
    print(f"    Pageouts: {vm_delta['pageout_delta_mb']:.1f} MB")

    return result


def main():
    from mlx_lm import load

    parser = argparse.ArgumentParser(description="SSD Layer LOD Phase 2: Scheduling Strategy")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    parser.add_argument("--strategy", default="all",
                        choices=["serial", "double-buf", "all"],
                        help="Scheduling strategy (or 'all' to compare)")
    parser.add_argument("--n-tokens", type=int, default=20)
    parser.add_argument("--pressure-gb", type=float, default=0)
    parser.add_argument("--save-dir", default="/tmp/ssd_lod_blocks")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 2: Scheduling Strategy Selection")
    print("=" * 60)

    env = get_environment_info()
    print(f"Environment: {env['chip']}, {env['memory_gb']} GB RAM, {env['available_gb']} GB available")

    print(f"\nLoading model: {args.model}")
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    blocks = get_model_blocks(model)
    print(f"Model has {len(blocks)} transformer blocks")

    save_dir = Path(args.save_dir)

    ballast = (None, None)
    regime = "unpressured"
    if args.pressure_gb > 0:
        regime = f"pressured ({args.pressure_gb}GB available)"
        print(f"\nApplying memory pressure: target {args.pressure_gb} GB available")
        ballast = create_memory_pressure(args.pressure_gb)

    # Run baseline first
    print("\n--- Baseline (all-resident) ---")
    prompt = "The transformer architecture has revolutionized natural language processing because"
    tokens = tokenizer.encode(prompt)

    baseline_times = []
    for i in range(args.n_tokens):
        t0 = time.perf_counter()
        input_ids = mx.array([tokens[-1:]])
        logits = model(input_ids, cache=None)
        next_token = mx.argmax(logits[0, -1, :]).item()  # .item() forces eval
        tokens.append(next_token)
        baseline_times.append((time.perf_counter() - t0) * 1000)

    steady_baseline = baseline_times[2:] if len(baseline_times) > 2 else baseline_times
    baseline_ms = sum(steady_baseline) / len(steady_baseline)
    print(f"  Baseline: {1000/baseline_ms:.1f} tok/s ({baseline_ms:.1f} ms/tok)")

    strategies = ["serial", "double-buf"] if args.strategy == "all" else [args.strategy]
    results = []

    for strategy in strategies:
        result = run_strategy_benchmark(
            model, tokenizer, blocks, strategy, args.n_tokens, save_dir,
        )
        result["regime"] = regime
        result["baseline_ms"] = round(baseline_ms, 1)
        result["slowdown"] = round(result["avg_tok_ms"] / baseline_ms, 1) if baseline_ms > 0 else 0
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY ({regime})")
    print(f"{'='*60}")
    print(f"  Baseline: {1000/baseline_ms:.1f} tok/s ({baseline_ms:.1f} ms/tok)")
    print(f"  {'Strategy':<15} {'tok/s':>8} {'ms/tok':>8} {'slowdown':>10} {'load p50':>10} {'wait p50':>10}")
    print(f"  {'-'*65}")
    for r in results:
        print(f"  {r['strategy']:<15} {r['tok_s']:>7.2f} {r['avg_tok_ms']:>7.0f} {r['slowdown']:>9.1f}x {r['load_p50_ms']:>9.1f}ms {r['wait_p50_ms']:>9.1f}ms")

    # Select winner
    if len(results) > 1:
        winner = min(results, key=lambda r: r["avg_tok_ms"])
        print(f"\n  Winner: {winner['strategy']} ({winner['tok_s']:.2f} tok/s)")
        improvement = (1 - winner["avg_tok_ms"] / max(r["avg_tok_ms"] for r in results)) * 100
        print(f"  Improvement: {improvement:.1f}% faster than slowest")

    log_experiment(
        experiment_name=f"ssd_lod_scheduler_{regime.replace(' ', '_')}",
        phase="scheduler",
        config={
            "model": args.model,
            "n_tokens": args.n_tokens,
            "regime": regime,
            "strategies": strategies,
        },
        results={
            "baseline_ms": round(baseline_ms, 1),
            "benchmarks": results,
        },
    )

    del ballast, model, tokenizer
    print(f"\nResults logged to experiments.jsonl")


if __name__ == "__main__":
    main()
