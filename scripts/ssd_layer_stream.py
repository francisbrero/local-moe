"""
Phase 1: Layer Streaming Prototype for SSD Layer LOD experiment.

Measures the performance impact of streaming transformer blocks from disk
vs keeping them all resident in RAM. Uses a custom forward pass that loads
each streaming block on-demand and evicts it after use, simulating the
real layer-streaming architecture.

Usage:
    uv run python scripts/ssd_layer_stream.py [--mode MODE] [--pressure-gb GB] [--n-tokens N]

Modes:
    all-resident   — Normal MLX inference, all blocks in RAM (baseline)
    stream-middle  — First/last 10% resident, middle 80% streamed from disk
    stream-all     — Everything streamed from disk (worst case)
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

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
    if hasattr(model, "layers"):
        return list(model.layers)
    raise ValueError("Could not find transformer layers in model")


def get_inner_model(model):
    """Get the inner model (Qwen2Model) from the causal LM wrapper."""
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
    return save_dir


def load_block_from_disk(block, block_idx: int, save_dir: Path):
    """Load a block's weights from disk and swap into the model."""
    block_file = save_dir / f"block_{block_idx:03d}.npz"
    data = np.load(block_file)

    for name, module in block.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            w_key = f"{name}.weight"
            s_key = f"{name}.scales"
            b_key = f"{name}.biases"
            if w_key in data:
                module.weight = mx.array(data[w_key])
                module.scales = mx.array(data[s_key])
                if b_key in data:
                    module.biases = mx.array(data[b_key])
    mx.eval(block.parameters())


def evict_block(block):
    """Replace block weights with tiny placeholders to free memory."""
    for name, module in block.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            # Replace with 1-element arrays to free memory
            module.weight = mx.zeros((1,), dtype=mx.uint32)
            module.scales = mx.zeros((1,), dtype=mx.float16)
            if hasattr(module, "biases") and module.biases is not None:
                module.biases = mx.zeros((1,), dtype=mx.float16)
    mx.eval(block.parameters())


def streaming_forward(model, input_ids, blocks, streaming_set, save_dir, per_block_load_ms, evict=True):
    """Custom forward pass that streams blocks on-demand.

    For each layer:
    - If it's a streaming block: load from disk, run forward, optionally evict
    - If it's resident: just run forward normally

    Args:
        evict: If True, evict blocks after use (requires mx.eval barrier).
               If False, keep blocks in memory (measures pure load overhead).

    Returns logits.
    """
    inner = get_inner_model(model)

    # Embedding
    h = inner.embed_tokens(input_ids)

    # Attention mask (None for single-token, "causal" for multi-token)
    mask = None
    if h.shape[1] > 1:
        mask = "causal"

    # Process each layer
    for i, layer in enumerate(inner.layers):
        if i in streaming_set:
            # Load block weights from disk
            t_load = time.perf_counter()
            load_block_from_disk(blocks[i], i, save_dir)
            load_ms = (time.perf_counter() - t_load) * 1000
            per_block_load_ms.append(load_ms)

        # Forward through this layer
        h = layer(h, mask, None)  # cache=None for simplicity

        if evict and i in streaming_set:
            # Ensure computation is done before evicting
            mx.eval(h)
            # Evict to free memory
            evict_block(blocks[i])

    # Final norm + lm_head
    h = inner.norm(h)
    if hasattr(model, "args") and model.args.tie_word_embeddings:
        out = inner.embed_tokens.as_linear(h)
    else:
        out = model.lm_head(h)

    return out


def run_streaming_benchmark(
    model,
    tokenizer,
    blocks,
    mode: str,
    n_tokens: int,
    save_dir: Path,
    resident_first_pct: float = 0.1,
    resident_last_pct: float = 0.1,
) -> dict:
    """Run inference with specified streaming mode and measure performance."""
    n_blocks = len(blocks)
    n_first = int(n_blocks * resident_first_pct)
    n_last = int(n_blocks * resident_last_pct)

    evict = True  # Whether to evict blocks after use (serializes GPU pipeline)

    if mode == "all-resident":
        streaming_indices = []
        resident_indices = list(range(n_blocks))
    elif mode in ("stream-middle", "stream-middle-no-evict"):
        resident_indices = list(range(n_first)) + list(range(n_blocks - n_last, n_blocks))
        streaming_indices = list(range(n_first, n_blocks - n_last))
        if mode == "stream-middle-no-evict":
            evict = False
    elif mode == "stream-all":
        streaming_indices = list(range(n_blocks))
        resident_indices = []
    else:
        raise ValueError(f"Unknown mode: {mode}")

    streaming_set = set(streaming_indices)

    print(f"\n  Mode: {mode}")
    print(f"  Resident blocks: {len(resident_indices)} (first {n_first} + last {n_last})")
    print(f"  Streaming blocks: {len(streaming_indices)}")

    # Save streaming blocks to disk
    if streaming_indices:
        print(f"  Saving {len(streaming_indices)} blocks to disk...")
        save_blocks_to_disk(blocks, streaming_indices, save_dir)

        # Evict streaming blocks to simulate memory-constrained state
        print(f"  Evicting {len(streaming_indices)} blocks from memory...")
        for idx in streaming_indices:
            evict_block(blocks[idx])
        gc.collect()
        mx.clear_cache()

    vm_before = get_vm_stat()
    rss_before = get_rss_mb()
    available_before = get_available_memory_gb()

    print(f"  RSS before: {rss_before:.1f} MB")
    print(f"  Available: {available_before:.1f} GB")
    print(f"  Generating {n_tokens} tokens...")

    prompt = "The transformer architecture has revolutionized natural language processing because"
    tokens = tokenizer.encode(prompt)

    per_token_ms = []
    per_block_load_ms = []

    for tok_i in range(n_tokens):
        t_tok_start = time.perf_counter()

        input_ids = mx.array([tokens[-1:]])  # shape [1, 1]

        if streaming_indices:
            # Custom forward pass with per-layer streaming
            logits = streaming_forward(
                model, input_ids, blocks, streaming_set, save_dir, per_block_load_ms,
                evict=evict,
            )
        else:
            # Normal forward pass (baseline)
            logits = model(input_ids, cache=None)

        next_token = mx.argmax(logits[0, -1, :]).item()  # .item() forces eval
        tokens.append(next_token)

        tok_ms = (time.perf_counter() - t_tok_start) * 1000
        per_token_ms.append(tok_ms)

        if tok_i < 5 or tok_i % 10 == 0:
            rss_now = get_rss_mb()
            print(f"    Token {tok_i}: {tok_ms:.0f} ms, RSS={rss_now:.0f} MB")

    vm_after = get_vm_stat()
    rss_after = get_rss_mb()
    peak_rss = get_peak_rss_mb()
    available_after = get_available_memory_gb()

    # Restore streaming blocks for next benchmark
    if streaming_indices:
        for idx in streaming_indices:
            load_block_from_disk(blocks[idx], idx, save_dir)

    # Compute statistics (skip first 2 warmup tokens)
    steady_times = per_token_ms[2:] if len(per_token_ms) > 2 else per_token_ms
    avg_tok_ms = sum(steady_times) / len(steady_times) if steady_times else 0
    tok_s = 1000 / avg_tok_ms if avg_tok_ms > 0 else 0

    if per_block_load_ms:
        load_p50 = float(np.percentile(per_block_load_ms, 50))
        load_p95 = float(np.percentile(per_block_load_ms, 95))
        load_p99 = float(np.percentile(per_block_load_ms, 99))
    else:
        load_p50 = load_p95 = load_p99 = 0.0

    vm_delta = vm_stat_delta(vm_before, vm_after)

    result = {
        "mode": mode,
        "n_tokens": n_tokens,
        "n_resident": len(resident_indices),
        "n_streaming": len(streaming_indices),
        "tok_s": round(tok_s, 2),
        "avg_tok_ms": round(avg_tok_ms, 1),
        "peak_rss_mb": round(peak_rss, 1),
        "rss_delta_mb": round(rss_after - rss_before, 1),
        "block_latency_p50_ms": round(load_p50, 1),
        "block_latency_p95_ms": round(load_p95, 1),
        "block_latency_p99_ms": round(load_p99, 1),
        "pageout_delta_mb": vm_delta["pageout_delta_mb"],
        "pagein_delta_mb": vm_delta["pagein_delta_mb"],
        "available_before_gb": round(available_before, 2),
        "available_after_gb": round(available_after, 2),
    }

    print(f"\n  Results:")
    print(f"    tok/s: {tok_s:.2f}")
    print(f"    Avg token time: {avg_tok_ms:.1f} ms")
    print(f"    Peak RSS: {peak_rss:.1f} MB")
    if per_block_load_ms:
        print(f"    Block load p50/p95/p99: {load_p50:.1f}/{load_p95:.1f}/{load_p99:.1f} ms")
    print(f"    Pageouts: {vm_delta['pageout_delta_mb']:.1f} MB")

    return result


def main():
    from mlx_lm import load

    parser = argparse.ArgumentParser(description="SSD Layer LOD Phase 1: Streaming Prototype")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    parser.add_argument("--mode", default="all",
                        choices=["all-resident", "stream-middle", "stream-middle-no-evict", "stream-all", "all"],
                        help="Streaming mode (or 'all' to run all modes)")
    parser.add_argument("--n-tokens", type=int, default=20,
                        help="Number of tokens to generate per mode")
    parser.add_argument("--pressure-gb", type=float, default=0,
                        help="Target available GB for pressure regime (0 = no pressure)")
    parser.add_argument("--save-dir", default="/tmp/ssd_lod_blocks",
                        help="Directory to save block weights for streaming")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 1: Layer Streaming Prototype")
    print("=" * 60)

    env = get_environment_info()
    print(f"Environment: {env['chip']}, {env['memory_gb']} GB RAM, {env['available_gb']} GB available")

    print(f"\nLoading model: {args.model}")
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())

    blocks = get_model_blocks(model)
    n_blocks = len(blocks)
    print(f"Model has {n_blocks} transformer blocks")

    save_dir = Path(args.save_dir)

    # Apply memory pressure if requested
    ballast = (None, None)
    regime = "unpressured"
    if args.pressure_gb > 0:
        regime = f"pressured ({args.pressure_gb}GB available)"
        print(f"\nApplying memory pressure: target {args.pressure_gb} GB available")
        ballast = create_memory_pressure(args.pressure_gb)

    modes = ["all-resident", "stream-middle", "stream-middle-no-evict", "stream-all"] if args.mode == "all" else [args.mode]
    results = []

    for mode in modes:
        result = run_streaming_benchmark(
            model, tokenizer, blocks, mode, args.n_tokens, save_dir,
            resident_first_pct=0.1, resident_last_pct=0.1,
        )
        result["regime"] = regime
        results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY ({regime})")
    print(f"{'='*60}")
    print(f"{'Mode':<20} {'tok/s':>8} {'RSS':>8} {'Load p50':>10} {'Load p95':>10}")
    print("-" * 60)
    for r in results:
        print(f"  {r['mode']:<18} {r['tok_s']:>7.2f} {r['peak_rss_mb']:>7.0f}M {r['block_latency_p50_ms']:>9.1f}ms {r['block_latency_p95_ms']:>9.1f}ms")

    # Evaluate gates
    baseline = next((r for r in results if r["mode"] == "all-resident"), None)
    stream_middle = next((r for r in results if r["mode"] == "stream-middle"), None)

    slowdown = float("inf")
    if baseline and stream_middle:
        slowdown = stream_middle["avg_tok_ms"] / baseline["avg_tok_ms"] if baseline["avg_tok_ms"] > 0 else float("inf")
        print(f"\n  Stream-middle vs all-resident: {slowdown:.1f}x slowdown")
        if args.pressure_gb > 0:
            gate = slowdown < 3.0
            print(f"  Gate (<3x under pressure): {'PASS' if gate else 'FAIL'}")
        else:
            gate = slowdown < 1.5
            print(f"  Gate (<1.5x unpressured): {'PASS' if gate else 'FAIL'}")

    # Log
    log_experiment(
        experiment_name=f"ssd_lod_stream_7b_{regime.replace(' ', '_')}",
        phase="layer_stream",
        config={
            "model": args.model,
            "n_tokens": args.n_tokens,
            "regime": regime,
            "pressure_gb": args.pressure_gb,
        },
        results={
            "benchmarks": results,
            "pass_fail": "PASS" if baseline and stream_middle and (
                (args.pressure_gb > 0 and slowdown < 3.0) or
                (args.pressure_gb == 0 and slowdown < 1.5)
            ) else "NEEDS_REVIEW",
        },
    )

    # Cleanup
    del ballast
    del model, tokenizer
    print(f"\nResults logged to experiments.jsonl")


if __name__ == "__main__":
    main()
