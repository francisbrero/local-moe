"""
expert_stream_synthetic.py — Phase 4a: Synthetic expert streaming microbenchmark.

Validates:
1. I/O streaming path overhead (in-memory vs SSD-streamed dequant+GEMM)
2. MLX zero-copy behavior with mmap-backed tensors
3. Per-expert and per-layer latency (p50/p95)

Usage:
    uv run python scripts/expert_stream_synthetic.py [--expert-size-mb SIZE] [--layers N] [--bits BITS]
"""

import argparse
import mmap
import os
import statistics
import struct
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from experiment_utils import (
    get_available_memory_gb,
    get_environment_info,
    get_rss_mb,
    get_vm_stat,
    log_experiment,
    vm_stat_delta,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_EXPERT_SIZE_MB = 4  # Per-expert quantized size
DEFAULT_HIDDEN_DIM = 2048
DEFAULT_FFN_DIM = 5632  # Typical MoE FFN dimension
DEFAULT_N_LAYERS = 48
DEFAULT_K_ACTIVE = 4
DEFAULT_N_SIMULATED_LAYERS = 100  # Simulated layer passes
DEFAULT_BITS = 4
ZERO_COPY_ITERATIONS = 100


# ---------------------------------------------------------------------------
# Expert tensor creation
# ---------------------------------------------------------------------------


def create_expert_file(
    n_experts: int,
    hidden_dim: int,
    ffn_dim: int,
    bits: int,
    path: Path,
) -> tuple[Path, int]:
    """Create a file of fake quantized expert tensors.

    Each expert has 3 matrices: gate_proj, up_proj, down_proj.
    Stored as packed quantized values (simulating GGUF/safetensors layout).

    Returns (filepath, bytes_per_expert).
    """
    # Params per expert: 3 * hidden * ffn
    params_per_expert = 3 * hidden_dim * ffn_dim
    bytes_per_expert = params_per_expert * bits // 8

    total_bytes = n_experts * bytes_per_expert
    print(f"  Creating expert file: {n_experts} experts x {bytes_per_expert / (1024**2):.2f} MB = {total_bytes / (1024**3):.2f} GB")

    # Write random data (simulating quantized weights)
    chunk_size = min(64 * 1024 * 1024, bytes_per_expert)
    with open(path, "wb") as f:
        written = 0
        while written < total_bytes:
            to_write = min(chunk_size, total_bytes - written)
            f.write(os.urandom(to_write))
            written += to_write

    os.sync()
    return path, bytes_per_expert


def load_expert_from_memory(
    data: bytes, hidden_dim: int, ffn_dim: int, bits: int
) -> mx.array:
    """Simulate loading and dequantizing an expert from in-memory data.

    Wraps raw bytes as uint8 MLX array, then simulates dequant to float16.
    """
    arr = mx.array(np.frombuffer(data, dtype=np.uint8))
    # Simulate dequantization: cast to float16 and scale
    # Real dequant would unpack nibbles, apply scales/zeros
    dequant = arr.astype(mx.float16) * (1.0 / 127.0)
    return dequant


def compute_matmul_dim(bytes_per_expert: int) -> int:
    """Compute the largest square matrix dimension that fits in an expert's bytes."""
    # After dequant from uint8 to float16, each byte becomes one float16 element
    n_elements = bytes_per_expert
    dim = int(n_elements**0.5)
    return dim


def expert_gemm(expert_weights: mx.array, input_tensor: mx.array, dim: int) -> mx.array:
    """Simulate expert FFN: simple matmul (not a real FFN, but measures compute cost)."""
    weight_2d = expert_weights[: dim * dim].reshape(dim, dim)
    result = input_tensor @ weight_2d
    mx.eval(result)
    return result


# ---------------------------------------------------------------------------
# Zero-copy test
# ---------------------------------------------------------------------------


def test_zero_copy(expert_file: Path, bytes_per_expert: int) -> dict:
    """Test if MLX materializes a copy when wrapping mmap'd data.

    Runs repeated wrap -> dequant -> matmul -> discard loop and tracks
    RSS, Metal peak memory, and allocation growth.
    """
    print(f"\n--- Zero-copy test ({ZERO_COPY_ITERATIONS} iterations) ---")

    fd = os.open(str(expert_file), os.O_RDONLY)
    mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)

    dim = compute_matmul_dim(bytes_per_expert)
    input_tensor = mx.random.normal((1, dim))
    mx.eval(input_tensor)

    # Baseline measurements
    rss_before = get_rss_mb()
    try:
        mx.metal.reset_peak_memory()
    except AttributeError:
        pass
    metal_before = None
    try:
        metal_before = mx.metal.get_active_memory() / (1024 * 1024)
    except AttributeError:
        pass

    rss_samples = []
    metal_samples = []

    for i in range(ZERO_COPY_ITERATIONS):
        # Read expert directly from mmap without intermediate bytes copy
        offset = (i % (len(mm) // bytes_per_expert)) * bytes_per_expert
        expert_np = np.frombuffer(
            mm, dtype=np.uint8, count=bytes_per_expert, offset=offset
        )

        # Wrap as MLX array (this is the zero-copy path we're testing)
        arr = mx.array(expert_np)

        # Dequant
        dequant = arr.astype(mx.float16) * (1.0 / 127.0)

        # Matmul
        weight_2d = dequant[: dim * dim].reshape(dim, dim)
        result = input_tensor @ weight_2d
        mx.eval(result)

        # Discard (must delete numpy view before mmap can close)
        del arr, dequant, weight_2d, result, expert_np

        # Sample memory
        if i % 10 == 0:
            rss_samples.append(get_rss_mb())
            try:
                metal_samples.append(mx.metal.get_active_memory() / (1024 * 1024))
            except AttributeError:
                pass

    rss_after = get_rss_mb()
    metal_after = None
    try:
        metal_after = mx.metal.get_active_memory() / (1024 * 1024)
    except AttributeError:
        pass
    metal_peak = None
    try:
        metal_peak = mx.metal.get_peak_memory() / (1024 * 1024)
    except AttributeError:
        try:
            metal_peak = mx.get_peak_memory() / (1024 * 1024)
        except AttributeError:
            pass

    mm.close()
    os.close(fd)

    rss_growth = rss_after - rss_before
    rss_growth_pct = (rss_growth / (bytes_per_expert / (1024 * 1024))) * 100 if bytes_per_expert > 0 else 0

    # Check if growth stays flat (< 5% of expert size across iterations)
    rss_range = max(rss_samples) - min(rss_samples) if rss_samples else 0
    metal_range = max(metal_samples) - min(metal_samples) if metal_samples else 0

    is_zero_copy = rss_growth_pct < 5 and rss_range < bytes_per_expert / (1024 * 1024) * 0.05

    result = {
        "rss_before_mb": round(rss_before, 1),
        "rss_after_mb": round(rss_after, 1),
        "rss_growth_mb": round(rss_growth, 1),
        "rss_growth_pct_of_expert": round(rss_growth_pct, 1),
        "rss_range_mb": round(rss_range, 1),
        "metal_before_mb": round(metal_before, 1) if metal_before else None,
        "metal_after_mb": round(metal_after, 1) if metal_after else None,
        "metal_peak_mb": round(metal_peak, 1) if metal_peak else None,
        "metal_range_mb": round(metal_range, 1) if metal_samples else None,
        "is_zero_copy": is_zero_copy,
        "iterations": ZERO_COPY_ITERATIONS,
    }

    verdict = "ZERO-COPY" if is_zero_copy else "COPIES DETECTED"
    print(f"  RSS growth: {rss_growth:.1f} MB ({rss_growth_pct:.1f}% of expert size)")
    print(f"  RSS range across iterations: {rss_range:.1f} MB")
    if metal_peak:
        print(f"  Metal peak: {metal_peak:.1f} MB")
    print(f"  Verdict: {verdict}")

    return result


# ---------------------------------------------------------------------------
# Streaming benchmark
# ---------------------------------------------------------------------------


def bench_in_memory_expert_gemm(
    expert_data: list[bytes],
    hidden_dim: int,
    ffn_dim: int,
    bits: int,
    n_layers: int,
    k_active: int,
) -> dict:
    """Benchmark dequant+GEMM with experts fully in memory."""
    dim = compute_matmul_dim(len(expert_data[0]))
    input_tensor = mx.random.normal((1, dim))
    mx.eval(input_tensor)

    n_experts = len(expert_data)
    latencies = []
    rng = np.random.default_rng(42)

    for layer in range(n_layers):
        chosen = rng.choice(n_experts, size=k_active, replace=False)
        for expert_idx in chosen:
            t0 = time.perf_counter()

            # Load from memory
            data = expert_data[expert_idx]
            arr = mx.array(np.frombuffer(data, dtype=np.uint8))
            dequant = arr.astype(mx.float16) * (1.0 / 127.0)

            # GEMM
            weight_2d = dequant[: dim * dim].reshape(dim, dim)
            result = input_tensor @ weight_2d
            mx.eval(result)

            elapsed = time.perf_counter() - t0
            latencies.append(elapsed * 1000)  # ms

            del arr, dequant, weight_2d, result

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    return {
        "mode": "in_memory",
        "latency_p50_ms": round(latencies_sorted[n // 2], 4) if n > 0 else 0,
        "latency_p95_ms": round(latencies_sorted[int(n * 0.95)], 4) if n > 0 else 0,
        "latency_mean_ms": round(statistics.mean(latencies), 4) if n > 0 else 0,
        "total_expert_calls": len(latencies),
        "total_time_ms": round(sum(latencies), 2),
    }


def bench_streamed_expert_gemm(
    expert_file: Path,
    bytes_per_expert: int,
    n_experts: int,
    hidden_dim: int,
    ffn_dim: int,
    bits: int,
    n_layers: int,
    k_active: int,
) -> dict:
    """Benchmark dequant+GEMM with experts streamed from SSD."""
    dim = compute_matmul_dim(bytes_per_expert)
    input_tensor = mx.random.normal((1, dim))
    mx.eval(input_tensor)

    fd = os.open(str(expert_file), os.O_RDONLY)
    latencies = []
    load_latencies = []
    compute_latencies = []
    rng = np.random.default_rng(42)

    try:
        for layer in range(n_layers):
            chosen = rng.choice(n_experts, size=k_active, replace=False)
            for expert_idx in chosen:
                t_total = time.perf_counter()

                # Read from SSD
                t_load = time.perf_counter()
                offset = expert_idx * bytes_per_expert
                data = os.pread(fd, bytes_per_expert, offset)
                load_time = time.perf_counter() - t_load

                # Dequant + GEMM
                t_compute = time.perf_counter()
                arr = mx.array(np.frombuffer(data, dtype=np.uint8))
                dequant = arr.astype(mx.float16) * (1.0 / 127.0)
                weight_2d = dequant[: dim * dim].reshape(dim, dim)
                result = input_tensor @ weight_2d
                mx.eval(result)
                compute_time = time.perf_counter() - t_compute

                total_time = time.perf_counter() - t_total
                latencies.append(total_time * 1000)
                load_latencies.append(load_time * 1000)
                compute_latencies.append(compute_time * 1000)

                del arr, dequant, weight_2d, result
    finally:
        os.close(fd)

    lat_sorted = sorted(latencies)
    load_sorted = sorted(load_latencies)
    compute_sorted = sorted(compute_latencies)
    n = len(lat_sorted)

    return {
        "mode": "streamed",
        "latency_p50_ms": round(lat_sorted[n // 2], 4) if n > 0 else 0,
        "latency_p95_ms": round(lat_sorted[int(n * 0.95)], 4) if n > 0 else 0,
        "latency_mean_ms": round(statistics.mean(latencies), 4) if n > 0 else 0,
        "load_p50_ms": round(load_sorted[n // 2], 4) if n > 0 else 0,
        "load_p95_ms": round(load_sorted[int(n * 0.95)], 4) if n > 0 else 0,
        "compute_p50_ms": round(compute_sorted[n // 2], 4) if n > 0 else 0,
        "compute_p95_ms": round(compute_sorted[int(n * 0.95)], 4) if n > 0 else 0,
        "total_expert_calls": len(latencies),
        "total_time_ms": round(sum(latencies), 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Phase 4a: Synthetic expert streaming")
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--ffn-dim", type=int, default=DEFAULT_FFN_DIM)
    parser.add_argument("--n-experts", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=DEFAULT_N_SIMULATED_LAYERS)
    parser.add_argument("--k-active", type=int, default=DEFAULT_K_ACTIVE)
    parser.add_argument("--bits", type=int, default=DEFAULT_BITS, choices=[2, 3, 4])
    args = parser.parse_args()

    env = get_environment_info()
    print(f"Phase 4a: Synthetic Expert Streaming Microbenchmark")
    print(f"Hardware: {env['chip']}, {env['memory_gb']} GB")
    print(f"Available: {get_available_memory_gb():.1f} GB")
    print(f"Config: hidden={args.hidden_dim}, ffn={args.ffn_dim}, experts={args.n_experts}, K={args.k_active}, bits={args.bits}")

    # Create expert file
    tmpdir = tempfile.mkdtemp(prefix="expert_stream_")
    expert_file = Path(tmpdir) / "experts.bin"
    expert_file, bytes_per_expert = create_expert_file(
        args.n_experts, args.hidden_dim, args.ffn_dim, args.bits, expert_file
    )
    print(f"  Bytes per expert: {bytes_per_expert / (1024**2):.2f} MB")

    all_results = {}

    # 1. Zero-copy test
    zero_copy = test_zero_copy(expert_file, bytes_per_expert)
    all_results["zero_copy"] = zero_copy

    # 2. In-memory baseline
    print(f"\n--- In-memory dequant+GEMM baseline ({args.n_layers} layers, K={args.k_active}) ---")
    # Load all experts into memory
    expert_data = []
    with open(expert_file, "rb") as f:
        for i in range(args.n_experts):
            expert_data.append(f.read(bytes_per_expert))

    rss_before = get_rss_mb()
    vm_before = get_vm_stat()
    in_memory = bench_in_memory_expert_gemm(
        expert_data, args.hidden_dim, args.ffn_dim, args.bits, args.n_layers, args.k_active
    )
    vm_after = get_vm_stat()
    in_memory["rss_mb"] = round(get_rss_mb(), 1)
    in_memory["rss_delta_mb"] = round(get_rss_mb() - rss_before, 1)
    in_memory["vm_stat_delta"] = vm_stat_delta(vm_before, vm_after)
    all_results["in_memory"] = in_memory
    print(f"  p50: {in_memory['latency_p50_ms']:.3f} ms, p95: {in_memory['latency_p95_ms']:.3f} ms")
    print(f"  Total: {in_memory['total_time_ms']:.1f} ms for {in_memory['total_expert_calls']} expert calls")

    # Free in-memory experts
    del expert_data

    # 3. SSD-streamed benchmark
    print(f"\n--- SSD-streamed dequant+GEMM ({args.n_layers} layers, K={args.k_active}) ---")
    rss_before = get_rss_mb()
    vm_before = get_vm_stat()
    streamed = bench_streamed_expert_gemm(
        expert_file, bytes_per_expert, args.n_experts,
        args.hidden_dim, args.ffn_dim, args.bits, args.n_layers, args.k_active,
    )
    vm_after = get_vm_stat()
    streamed["rss_mb"] = round(get_rss_mb(), 1)
    streamed["rss_delta_mb"] = round(get_rss_mb() - rss_before, 1)
    streamed["vm_stat_delta"] = vm_stat_delta(vm_before, vm_after)
    all_results["streamed"] = streamed
    print(f"  p50: {streamed['latency_p50_ms']:.3f} ms, p95: {streamed['latency_p95_ms']:.3f} ms")
    print(f"  Load p50: {streamed['load_p50_ms']:.3f} ms, Compute p50: {streamed['compute_p50_ms']:.3f} ms")
    print(f"  Total: {streamed['total_time_ms']:.1f} ms for {streamed['total_expert_calls']} expert calls")

    # 4. Comparison
    overhead_ratio = streamed["latency_mean_ms"] / in_memory["latency_mean_ms"] if in_memory["latency_mean_ms"] > 0 else float("inf")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<30} {'In-Memory':>12} {'Streamed':>12} {'Ratio':>8}")
    print(f"{'-'*30} {'-'*12} {'-'*12} {'-'*8}")
    print(f"{'Latency p50 (ms)':<30} {in_memory['latency_p50_ms']:>12.3f} {streamed['latency_p50_ms']:>12.3f} {streamed['latency_p50_ms']/in_memory['latency_p50_ms'] if in_memory['latency_p50_ms'] > 0 else 0:>8.2f}x")
    print(f"{'Latency p95 (ms)':<30} {in_memory['latency_p95_ms']:>12.3f} {streamed['latency_p95_ms']:>12.3f} {streamed['latency_p95_ms']/in_memory['latency_p95_ms'] if in_memory['latency_p95_ms'] > 0 else 0:>8.2f}x")
    print(f"{'Total time (ms)':<30} {in_memory['total_time_ms']:>12.1f} {streamed['total_time_ms']:>12.1f} {overhead_ratio:>8.2f}x")
    print(f"{'RSS delta (MB)':<30} {in_memory['rss_delta_mb']:>12.1f} {streamed['rss_delta_mb']:>12.1f}")
    print(f"{'Pageout (MB)':<30} {in_memory['vm_stat_delta']['pageout_delta_mb']:>12.1f} {streamed['vm_stat_delta']['pageout_delta_mb']:>12.1f}")

    print(f"\nOverhead ratio: {overhead_ratio:.2f}x")
    if overhead_ratio <= 3:
        print(f"  PASS: Within 3x threshold for prototype viability")
    else:
        print(f"  FAIL: Exceeds 3x threshold")

    print(f"\nZero-copy: {'YES' if zero_copy['is_zero_copy'] else 'NO'}")
    if zero_copy["is_zero_copy"]:
        print(f"  Recommended Phase 4b path: MLX mmap-based tensor wrapping")
    else:
        print(f"  Recommended Phase 4b path: C/Metal staging buffers")

    # Cleanup
    expert_file.unlink()
    os.rmdir(tmpdir)

    # Log
    log_experiment(
        experiment_name=f"expert_stream_synthetic_{args.bits}bit",
        phase="expert_stream_synthetic",
        config={
            "hidden_dim": args.hidden_dim,
            "ffn_dim": args.ffn_dim,
            "n_experts": args.n_experts,
            "n_layers": args.n_layers,
            "k_active": args.k_active,
            "bits": args.bits,
            "bytes_per_expert": bytes_per_expert,
        },
        results={
            "zero_copy": zero_copy,
            "in_memory": in_memory,
            "streamed": streamed,
            "overhead_ratio": round(overhead_ratio, 3),
            "verdict": "viable" if overhead_ratio <= 3 else "too_slow",
            "recommended_path": "mlx_mmap" if zero_copy["is_zero_copy"] else "c_metal_staging",
        },
        env=env,
    )


if __name__ == "__main__":
    main()
