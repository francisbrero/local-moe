"""
Phase 2b: Synthetic 72B-Shaped Streaming Benchmark.

Simulates the 72B streaming workload by creating synthetic block files
matching the actual 72B block size and accessing them sequentially under
memory pressure. Measures page cache behavior, per-block latency, and
thrashing detection without requiring the actual 72B model.

Usage:
    uv run python scripts/ssd_synthetic_stream.py [--n-blocks N] [--block-size-mb MB] [--pressure-gb GB] [--iterations N]
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

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


def create_synthetic_blocks(n_blocks: int, block_size_mb: float, save_dir: Path):
    """Create synthetic block files with random data."""
    save_dir.mkdir(parents=True, exist_ok=True)
    block_size_bytes = int(block_size_mb * 1024 * 1024)

    for i in range(n_blocks):
        block_file = save_dir / f"synth_block_{i:03d}.bin"
        if block_file.exists() and block_file.stat().st_size == block_size_bytes:
            continue
        print(f"  Creating block {i}/{n_blocks} ({block_size_mb:.0f} MB)...", end="\r")
        # Write random data in chunks to avoid huge memory allocation
        chunk_size = 64 * 1024 * 1024  # 64 MB chunks
        with open(block_file, "wb") as f:
            remaining = block_size_bytes
            while remaining > 0:
                size = min(chunk_size, remaining)
                f.write(os.urandom(size))
                remaining -= size
    print(f"  Created {n_blocks} blocks ({block_size_mb:.0f} MB each)             ")


def read_block_pread(block_idx: int, save_dir: Path, buf: bytearray) -> float:
    """Read a block using pread into a pre-allocated buffer. Returns latency in ms."""
    block_file = save_dir / f"synth_block_{block_idx:03d}.bin"
    fd = os.open(str(block_file), os.O_RDONLY)
    try:
        t0 = time.perf_counter()
        total = len(buf)
        offset = 0
        while offset < total:
            chunk = min(total - offset, 64 * 1024 * 1024)  # 64 MB per pread call
            data = os.pread(fd, chunk, offset)
            if not data:
                raise IOError(f"pread returned 0 bytes at offset {offset}")
            buf[offset:offset + len(data)] = data
            offset += len(data)
        elapsed_ms = (time.perf_counter() - t0) * 1000
    finally:
        os.close(fd)
    return elapsed_ms


def run_synthetic_benchmark(
    n_blocks: int,
    block_size_mb: float,
    iterations: int,
    save_dir: Path,
) -> dict:
    """Run synthetic sequential streaming benchmark."""
    block_size_bytes = int(block_size_mb * 1024 * 1024)
    total_streaming_gb = n_blocks * block_size_mb / 1024

    print(f"\n  Streaming: {n_blocks} blocks × {block_size_mb:.0f} MB = {total_streaming_gb:.1f} GB")
    print(f"  Iterations: {iterations} (each reads all {n_blocks} blocks)")

    # Pre-allocate staging buffer
    buf = bytearray(block_size_bytes)

    vm_before = get_vm_stat()
    rss_before = get_rss_mb()
    available_before = get_available_memory_gb()

    print(f"  RSS: {rss_before:.0f} MB, Available: {available_before:.1f} GB")

    per_block_latencies = []  # (iteration, block_idx, latency_ms)
    per_iteration_ms = []

    for it in range(iterations):
        t_iter = time.perf_counter()

        for block_idx in range(n_blocks):
            latency_ms = read_block_pread(block_idx, save_dir, buf)
            per_block_latencies.append((it, block_idx, latency_ms))

        iter_ms = (time.perf_counter() - t_iter) * 1000
        per_iteration_ms.append(iter_ms)

        if it < 3 or it % 5 == 0:
            avail = get_available_memory_gb()
            rss = get_rss_mb()
            avg_lat = sum(l[2] for l in per_block_latencies[-n_blocks:]) / n_blocks
            print(f"    Iter {it}: {iter_ms:.0f} ms ({avg_lat:.0f} ms/block avg), "
                  f"RSS={rss:.0f} MB, Available={avail:.1f} GB")

    vm_after = get_vm_stat()
    rss_after = get_rss_mb()
    peak_rss = get_peak_rss_mb()

    # Compute statistics
    all_latencies = [l[2] for l in per_block_latencies]
    # Skip first iteration for warmup
    warmup_latencies = [l[2] for l in per_block_latencies if l[0] >= 1]

    # Per-iteration analysis
    steady_iters = per_iteration_ms[1:] if len(per_iteration_ms) > 1 else per_iteration_ms
    avg_iter_ms = sum(steady_iters) / len(steady_iters) if steady_iters else 0
    simulated_tok_ms = avg_iter_ms  # One iteration = one "token" (all blocks accessed once)

    # Cache warmup analysis: compare first iter to steady state
    first_iter_latencies = [l[2] for l in per_block_latencies if l[0] == 0]
    last_iter_latencies = [l[2] for l in per_block_latencies if l[0] == iterations - 1]

    first_avg = sum(first_iter_latencies) / len(first_iter_latencies) if first_iter_latencies else 0
    last_avg = sum(last_iter_latencies) / len(last_iter_latencies) if last_iter_latencies else 0
    warmup_speedup = first_avg / last_avg if last_avg > 0 else 0

    # Thrash detection: check if latencies are stable or oscillating
    iter_averages = []
    for it in range(iterations):
        iter_lats = [l[2] for l in per_block_latencies if l[0] == it]
        iter_averages.append(sum(iter_lats) / len(iter_lats))

    # Coefficient of variation of per-iteration averages (after warmup)
    steady_avgs = iter_averages[2:] if len(iter_averages) > 2 else iter_averages
    if steady_avgs and len(steady_avgs) > 1:
        cv = float(np.std(steady_avgs) / np.mean(steady_avgs))
    else:
        cv = 0.0
    thrashing = cv > 0.3  # >30% variation suggests thrashing

    vm_delta = vm_stat_delta(vm_before, vm_after)

    result = {
        "n_blocks": n_blocks,
        "block_size_mb": block_size_mb,
        "total_streaming_gb": round(total_streaming_gb, 1),
        "iterations": iterations,
        "simulated_tok_ms": round(simulated_tok_ms, 0),
        "simulated_tok_s": round(1000 / simulated_tok_ms, 2) if simulated_tok_ms > 0 else 0,
        "block_latency_p50_ms": round(float(np.percentile(warmup_latencies, 50)), 1) if warmup_latencies else 0,
        "block_latency_p95_ms": round(float(np.percentile(warmup_latencies, 95)), 1) if warmup_latencies else 0,
        "block_latency_p99_ms": round(float(np.percentile(warmup_latencies, 99)), 1) if warmup_latencies else 0,
        "first_iter_avg_ms": round(first_avg, 1),
        "last_iter_avg_ms": round(last_avg, 1),
        "warmup_speedup": round(warmup_speedup, 1),
        "latency_cv": round(cv, 3),
        "thrashing": thrashing,
        "peak_rss_mb": round(peak_rss, 1),
        "pageout_delta_mb": vm_delta["pageout_delta_mb"],
        "pagein_delta_mb": vm_delta["pagein_delta_mb"],
        "available_before_gb": round(available_before, 2),
        "available_after_gb": round(get_available_memory_gb(), 2),
    }

    print(f"\n  Results:")
    print(f"    Simulated tok/s: {result['simulated_tok_s']:.2f} ({simulated_tok_ms:.0f} ms/tok)")
    print(f"    Block latency p50/p95/p99: {result['block_latency_p50_ms']:.0f}/{result['block_latency_p95_ms']:.0f}/{result['block_latency_p99_ms']:.0f} ms")
    print(f"    Cache warmup: {first_avg:.0f} ms → {last_avg:.0f} ms ({warmup_speedup:.1f}x)")
    print(f"    Thrashing: {'YES' if thrashing else 'NO'} (CV={cv:.3f})")
    print(f"    Pageouts: {vm_delta['pageout_delta_mb']:.0f} MB, Pageins: {vm_delta['pagein_delta_mb']:.0f} MB")

    return result


def main():
    parser = argparse.ArgumentParser(description="SSD Layer LOD Phase 2b: Synthetic 72B Streaming")
    parser.add_argument("--n-blocks", type=int, default=64,
                        help="Number of streaming blocks (default: 64 from 8+8 Q4/64 Q2)")
    parser.add_argument("--block-size-mb", type=float, default=262,
                        help="Block size in MB (default: 262 for Q2 72B block)")
    parser.add_argument("--pressure-gb", type=float, default=0,
                        help="Target available GB (0 = no pressure)")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of sequential passes (simulated tokens)")
    parser.add_argument("--save-dir", default="/tmp/ssd_lod_synth_blocks")
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 2b: Synthetic 72B Streaming Benchmark")
    print("=" * 60)

    env = get_environment_info()
    print(f"Environment: {env['chip']}, {env['memory_gb']} GB RAM, {env['available_gb']} GB available")

    save_dir = Path(args.save_dir)

    print(f"\nCreating {args.n_blocks} synthetic blocks ({args.block_size_mb:.0f} MB each)...")
    create_synthetic_blocks(args.n_blocks, args.block_size_mb, save_dir)

    ballast = (None, None)
    regime = "unpressured"
    if args.pressure_gb > 0:
        regime = f"pressured ({args.pressure_gb}GB available)"
        print(f"\nApplying memory pressure: target {args.pressure_gb} GB available")
        ballast = create_memory_pressure(args.pressure_gb)

    result = run_synthetic_benchmark(
        args.n_blocks, args.block_size_mb, args.iterations, save_dir,
    )
    result["regime"] = regime

    # Gate check
    print(f"\n  Gate check (p95 < 50 ms): ", end="")
    if result["block_latency_p95_ms"] < 50:
        print("PASS")
        result["gate"] = "PASS"
    else:
        print(f"FAIL ({result['block_latency_p95_ms']:.0f} ms)")
        result["gate"] = "FAIL"

    log_experiment(
        experiment_name=f"ssd_lod_synth_72b_{regime.replace(' ', '_')}",
        phase="synthetic_stream",
        config={
            "n_blocks": args.n_blocks,
            "block_size_mb": args.block_size_mb,
            "iterations": args.iterations,
            "regime": regime,
            "pressure_gb": args.pressure_gb,
        },
        results=result,
    )

    del ballast
    print(f"\nResults logged to experiments.jsonl")


if __name__ == "__main__":
    main()
