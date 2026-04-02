"""
Phase 3: 72B Integration Test for SSD Layer LOD.

Downloads Qwen2.5-72B-Instruct-4bit and runs streaming inference with
the double-buffer scheduler. First/last N blocks stay resident in RAM,
middle blocks are streamed from disk on demand.

Usage:
    uv run python scripts/ssd_lod_72b_integration.py [--n-tokens N] [--resident-pct PCT]

This is the main event — running 72B on a 24 GB machine.
"""

import argparse
import gc
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.experiment_utils import (
    get_environment_info,
    get_rss_mb,
    get_peak_rss_mb,
    get_vm_stat,
    vm_stat_delta,
    log_experiment,
    get_available_memory_gb,
)


# ── Block utilities (shared with Phase 1/2 scripts) ──


def get_model_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "layers"):
        return list(model.layers)
    raise ValueError("Could not find transformer layers in model")


def get_inner_model(model):
    if hasattr(model, "model"):
        return model.model
    return model


def save_block_to_disk(block, idx, save_dir: Path):
    """Save a single block's weights to disk as .npz."""
    block_file = save_dir / f"block_{idx:03d}.npz"
    if block_file.exists():
        return
    flat = {}
    for name, module in block.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            flat[f"{name}.weight"] = np.array(module.weight)
            flat[f"{name}.scales"] = np.array(module.scales)
            if hasattr(module, "biases") and module.biases is not None:
                flat[f"{name}.biases"] = np.array(module.biases)
    np.savez(block_file, **flat)


def load_block_weights_from_disk(block_idx: int, save_dir: Path) -> dict:
    """Load block weights from disk into numpy dict (CPU I/O, no MLX)."""
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
    """Replace block weights with tiny placeholders to free memory."""
    for name, module in block.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            module.weight = mx.zeros((1,), dtype=mx.uint32)
            module.scales = mx.zeros((1,), dtype=mx.float16)
            if hasattr(module, "biases") and module.biases is not None:
                module.biases = mx.zeros((1,), dtype=mx.float16)
    mx.eval(block.parameters())


# ── Double-buffer streaming forward pass ──


def double_buffer_forward(model, input_ids, blocks, streaming_indices, save_dir, metrics, executor):
    """Forward pass with double-buffer prefetch for streaming blocks."""
    inner = get_inner_model(model)
    h = inner.embed_tokens(input_ids)
    mask = None if h.shape[1] == 1 else "causal"
    streaming_set = set(streaming_indices)

    stream_order = sorted(streaming_indices)
    next_stream = {}
    for j in range(len(stream_order) - 1):
        next_stream[stream_order[j]] = stream_order[j + 1]

    pending_future = None
    pending_idx = None

    if stream_order:
        first_idx = stream_order[0]
        pending_future = executor.submit(load_block_weights_from_disk, first_idx, save_dir)
        pending_idx = first_idx

    for i, layer in enumerate(inner.layers):
        if i in streaming_set:
            if pending_idx == i and pending_future is not None:
                t_wait = time.perf_counter()
                weights = pending_future.result()
                wait_ms = (time.perf_counter() - t_wait) * 1000
                metrics["wait_ms"].append(wait_ms)
                pending_future = None
                pending_idx = None
            else:
                t0 = time.perf_counter()
                weights = load_block_weights_from_disk(i, save_dir)
                load_ms = (time.perf_counter() - t0) * 1000
                metrics["wait_ms"].append(load_ms)

            t_swap = time.perf_counter()
            swap_block_weights(blocks[i], weights)
            metrics["swap_ms"].append((time.perf_counter() - t_swap) * 1000)

        h = layer(h, mask, None)

        if i in streaming_set:
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


# ── Main ──


def main():
    from mlx_lm import load

    parser = argparse.ArgumentParser(description="SSD Layer LOD Phase 3: 72B Integration")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-72B-Instruct-4bit")
    parser.add_argument("--n-tokens", type=int, default=50,
                        help="Number of tokens to generate")
    parser.add_argument("--resident-pct", type=float, default=0.1,
                        help="Fraction of blocks to keep resident at each end (default 0.1 = 10%%)")
    parser.add_argument("--save-dir", default="/tmp/ssd_lod_72b_blocks")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 3: 72B Integration Test — SSD Layer LOD")
    print("=" * 70)

    env = get_environment_info()
    print(f"Environment: {env['chip']}, {env['memory_gb']} GB RAM, {env['available_gb']} GB available")
    print(f"Model: {args.model}")

    # Step 1: Load the model (lazy — MLX uses mmap, no full materialization)
    print(f"\nStep 1: Loading model (lazy, via mmap)...")
    t_load_start = time.perf_counter()
    model, tokenizer = load(args.model)
    load_s = time.perf_counter() - t_load_start
    print(f"  Model loaded in {load_s:.0f}s (lazy — weights not yet in RAM)")

    blocks = get_model_blocks(model)
    n_blocks = len(blocks)
    print(f"  {n_blocks} transformer blocks")

    # Compute resident/streaming split
    n_first = int(n_blocks * args.resident_pct)
    n_last = int(n_blocks * args.resident_pct)
    n_first = max(n_first, 1)
    n_last = max(n_last, 1)

    resident_indices = list(range(n_first)) + list(range(n_blocks - n_last, n_blocks))
    resident_set = set(resident_indices)
    streaming_indices = [i for i in range(n_blocks) if i not in resident_set]

    print(f"  Split: first {n_first} + last {n_last} resident ({len(resident_indices)} blocks), "
          f"{len(streaming_indices)} streaming")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: Incrementally materialize, save, and evict blocks ONE AT A TIME.
    # This avoids ever having all 38 GB in memory simultaneously.
    # For each streaming block: eval its params → save to disk → evict (replace with placeholders)
    # For each resident block: eval its params → keep in memory
    print(f"\nStep 2: Incremental block processing ({n_blocks} blocks)...")
    print(f"  Streaming blocks will be saved to disk then evicted.")
    print(f"  Resident blocks will be materialized and kept in RAM.")

    t_save_start = time.perf_counter()
    block_size_mb = 0

    for idx in range(n_blocks):
        if idx in resident_set:
            # Resident: materialize and keep
            mx.eval(blocks[idx].parameters())
            tag = "resident"
        else:
            # Streaming: materialize → save → evict
            mx.eval(blocks[idx].parameters())
            save_block_to_disk(blocks[idx], idx, save_dir)
            evict_block(blocks[idx])
            gc.collect()
            tag = "saved+evicted"

        if idx % 10 == 0 or idx == n_blocks - 1:
            avail = get_available_memory_gb()
            rss = get_rss_mb()
            print(f"  Block {idx:>3d}/{n_blocks} [{tag:<15s}] RSS={rss:.0f} MB, Avail={avail:.1f} GB")

    # Also materialize non-block components (embeddings, lm_head, norm)
    inner = get_inner_model(model)
    mx.eval(inner.embed_tokens.parameters())
    mx.eval(inner.norm.parameters())
    if hasattr(model, "lm_head"):
        mx.eval(model.lm_head.parameters())

    save_s = time.perf_counter() - t_save_start
    print(f"  Processed all blocks in {save_s:.0f}s")

    mx.clear_cache()

    # Check block file sizes
    sample_file = save_dir / f"block_{streaming_indices[0]:03d}.npz"
    block_size_mb = sample_file.stat().st_size / (1024 * 1024)
    total_streaming_gb = block_size_mb * len(streaming_indices) / 1024
    print(f"  Block size: {block_size_mb:.0f} MB, Total streaming: {total_streaming_gb:.1f} GB")

    rss_after_setup = get_rss_mb()
    avail_after_setup = get_available_memory_gb()
    print(f"  RSS after setup: {rss_after_setup:.0f} MB")
    print(f"  Available after setup: {avail_after_setup:.1f} GB")

    # Step 5: Generate tokens with streaming
    print(f"\nStep 4: Generating {args.n_tokens} tokens with double-buffer streaming...")

    prompt = "The key innovation of the transformer architecture is"
    tokens = tokenizer.encode(prompt)
    print(f"  Prompt: '{prompt}' ({len(tokens)} tokens)")

    vm_before = get_vm_stat()
    per_token_ms = []
    all_wait_ms = []
    all_swap_ms = []
    avail_history = []
    rss_history = []

    executor = ThreadPoolExecutor(max_workers=1)

    for tok_i in range(args.n_tokens):
        t0 = time.perf_counter()
        input_ids = mx.array([tokens[-1:]])

        metrics = {"wait_ms": [], "swap_ms": []}
        logits = double_buffer_forward(
            model, input_ids, blocks, streaming_indices, save_dir, metrics, executor
        )
        next_token = mx.argmax(logits[0, -1, :]).item()
        tokens.append(next_token)

        tok_ms = (time.perf_counter() - t0) * 1000
        per_token_ms.append(tok_ms)
        all_wait_ms.extend(metrics["wait_ms"])
        all_swap_ms.extend(metrics["swap_ms"])

        avail_now = get_available_memory_gb()
        rss_now = get_rss_mb()
        avail_history.append(avail_now)
        rss_history.append(rss_now)

        if tok_i < 5 or tok_i % 10 == 0 or tok_i == args.n_tokens - 1:
            total_wait = sum(metrics["wait_ms"])
            total_swap = sum(metrics["swap_ms"])
            print(f"    Token {tok_i}: {tok_ms:.0f} ms "
                  f"(wait={total_wait:.0f} swap={total_swap:.0f}), "
                  f"RSS={rss_now:.0f} MB, Avail={avail_now:.1f} GB")

    executor.shutdown(wait=True)

    vm_after = get_vm_stat()
    vm_delta = vm_stat_delta(vm_before, vm_after)
    peak_rss = get_peak_rss_mb()

    # Decode generated text
    generated_text = tokenizer.decode(tokens[len(tokenizer.encode(prompt)):])
    print(f"\n  Generated text: '{generated_text[:200]}...'")

    # Step 6: Compute statistics
    steady = per_token_ms[2:]  # skip warmup
    avg_tok_ms = sum(steady) / len(steady) if steady else 0
    tok_s = 1000 / avg_tok_ms if avg_tok_ms > 0 else 0
    min_avail = min(avail_history) if avail_history else 0
    max_rss = max(rss_history) if rss_history else 0

    print(f"\n{'='*70}")
    print(f"RESULTS — 72B Integration Test")
    print(f"{'='*70}")
    print(f"  Model: {args.model}")
    print(f"  Blocks: {n_blocks} total, {len(resident_indices)} resident, {len(streaming_indices)} streaming")
    print(f"  Block size: {block_size_mb:.0f} MB, Streaming total: {total_streaming_gb:.1f} GB")
    print(f"  Tokens generated: {args.n_tokens}")
    print(f"")
    print(f"  tok/s: {tok_s:.3f} ({avg_tok_ms:.0f} ms/tok)")
    print(f"  Peak RSS: {peak_rss:.0f} MB")
    print(f"  Max RSS during gen: {max_rss:.0f} MB")
    print(f"  Min available during gen: {min_avail:.1f} GB")
    print(f"  Memory stable: {'YES' if min_avail > 1.0 else 'NO (below 1 GB!)'}")
    print(f"")

    if all_wait_ms:
        print(f"  Block wait p50/p95: {np.percentile(all_wait_ms, 50):.0f}/{np.percentile(all_wait_ms, 95):.0f} ms")
    if all_swap_ms:
        print(f"  Block swap p50/p95: {np.percentile(all_swap_ms, 50):.0f}/{np.percentile(all_swap_ms, 95):.0f} ms")

    print(f"  Pageouts: {vm_delta['pageout_delta_mb']:.0f} MB")
    print(f"  Pageins: {vm_delta['pagein_delta_mb']:.0f} MB")

    # Gate checks
    print(f"\n  Gate checks:")
    gate_tok_s = tok_s >= 0.1  # Very conservative floor
    gate_memory = min_avail > 1.0
    gate_stable = max(avail_history[-10:]) - min(avail_history[-10:]) < 2.0 if len(avail_history) >= 10 else True
    print(f"    tok/s >= 0.1: {'PASS' if gate_tok_s else 'FAIL'} ({tok_s:.3f})")
    print(f"    Available > 1 GB: {'PASS' if gate_memory else 'FAIL'} ({min_avail:.1f} GB)")
    print(f"    Memory stable (last 10 tok): {'PASS' if gate_stable else 'FAIL'}")
    overall = gate_tok_s and gate_memory and gate_stable
    print(f"    Overall: {'PASS' if overall else 'FAIL'}")

    # Log results
    log_experiment(
        experiment_name="ssd_lod_72b_integration",
        phase="integration_72b",
        config={
            "model": args.model,
            "n_tokens": args.n_tokens,
            "n_blocks": n_blocks,
            "n_resident": len(resident_indices),
            "n_streaming": len(streaming_indices),
            "resident_pct": args.resident_pct,
            "block_size_mb": round(block_size_mb, 1),
            "total_streaming_gb": round(total_streaming_gb, 1),
        },
        results={
            "tok_s": round(tok_s, 4),
            "avg_tok_ms": round(avg_tok_ms, 1),
            "peak_rss_mb": round(peak_rss, 1),
            "min_available_gb": round(min_avail, 2),
            "block_wait_p50_ms": round(float(np.percentile(all_wait_ms, 50)), 1) if all_wait_ms else 0,
            "block_wait_p95_ms": round(float(np.percentile(all_wait_ms, 95)), 1) if all_wait_ms else 0,
            "block_swap_p50_ms": round(float(np.percentile(all_swap_ms, 50)), 1) if all_swap_ms else 0,
            "pageout_delta_mb": vm_delta["pageout_delta_mb"],
            "pagein_delta_mb": vm_delta["pagein_delta_mb"],
            "memory_stable": gate_stable,
            "pass_fail": "PASS" if overall else "FAIL",
            "generated_text_preview": generated_text[:200],
        },
    )

    del model, tokenizer
    print(f"\nResults logged to experiments.jsonl")


if __name__ == "__main__":
    main()
