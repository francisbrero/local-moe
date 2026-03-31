"""
gpu_ssd_contention.py — Phase 3: Serial vs concurrent GPU/SSD access on M4.

Validates the reported 73% GPU throughput degradation when GPU compute
and SSD reads overlap on Apple Silicon unified memory.

Usage:
    uv run python scripts/gpu_ssd_contention.py [--matrix-size N] [--file-size-gb SIZE]
"""

import argparse
import os
import statistics
import tempfile
import threading
import time
from pathlib import Path

import mlx.core as mx

from experiment_utils import (
    get_available_memory_gb,
    get_environment_info,
    get_vm_stat,
    log_experiment,
    vm_stat_delta,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MATRIX_SIZE = 4096  # NxN matmul
DEFAULT_FILE_SIZE_GB = 2
CHUNK_SIZE_MB = 4
GPU_ITERATIONS = 50
SSD_ITERATIONS = 50
REPETITIONS = 3


# ---------------------------------------------------------------------------
# GPU benchmark
# ---------------------------------------------------------------------------


def bench_gpu_only(n: int, iterations: int) -> dict:
    """Benchmark GPU matmul throughput in isolation."""
    a = mx.random.normal((n, n))
    b = mx.random.normal((n, n))
    mx.eval(a, b)

    # Warmup
    for _ in range(3):
        c = a @ b
        mx.eval(c)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        c = a @ b
        mx.eval(c)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    # GFLOPS = 2*N^3 / time / 1e9
    flops_per_op = 2 * n * n * n
    mean_time = statistics.mean(times)
    gflops = flops_per_op / mean_time / 1e9

    return {
        "mean_time_ms": round(mean_time * 1000, 3),
        "std_time_ms": round(statistics.stdev(times) * 1000, 3) if len(times) > 1 else 0,
        "gflops": round(gflops, 1),
        "iterations": iterations,
    }


# ---------------------------------------------------------------------------
# SSD benchmark
# ---------------------------------------------------------------------------


def bench_ssd_only(filepath: Path, chunk_bytes: int, iterations: int) -> dict:
    """Benchmark SSD read throughput in isolation."""
    file_size = filepath.stat().st_size
    times = []
    bytes_read_total = 0

    fd = os.open(str(filepath), os.O_RDONLY)
    try:
        for i in range(iterations):
            offset = (i * chunk_bytes) % (file_size - chunk_bytes)
            t0 = time.perf_counter()
            data = os.pread(fd, chunk_bytes, offset)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            bytes_read_total += len(data)
    finally:
        os.close(fd)

    mean_time = statistics.mean(times)
    bandwidth = chunk_bytes / mean_time / (1024**3)

    return {
        "mean_time_ms": round(mean_time * 1000, 3),
        "bandwidth_gbps": round(bandwidth, 3),
        "bytes_read_gb": round(bytes_read_total / (1024**3), 3),
        "iterations": iterations,
    }


# ---------------------------------------------------------------------------
# Serial pipeline
# ---------------------------------------------------------------------------


def bench_serial(
    n: int, filepath: Path, chunk_bytes: int, iterations: int
) -> dict:
    """Benchmark serial: SSD read then GPU compute, alternating."""
    a = mx.random.normal((n, n))
    b = mx.random.normal((n, n))
    mx.eval(a, b)

    file_size = filepath.stat().st_size
    fd = os.open(str(filepath), os.O_RDONLY)

    gpu_times = []
    ssd_times = []

    try:
        for i in range(iterations):
            # SSD read
            offset = (i * chunk_bytes) % (file_size - chunk_bytes)
            t0 = time.perf_counter()
            data = os.pread(fd, chunk_bytes, offset)
            ssd_time = time.perf_counter() - t0
            ssd_times.append(ssd_time)

            # GPU compute
            t0 = time.perf_counter()
            c = a @ b
            mx.eval(c)
            gpu_time = time.perf_counter() - t0
            gpu_times.append(gpu_time)
    finally:
        os.close(fd)

    flops_per_op = 2 * n * n * n
    mean_gpu = statistics.mean(gpu_times)
    mean_ssd = statistics.mean(ssd_times)
    total_per_iter = mean_gpu + mean_ssd

    return {
        "gpu_mean_ms": round(mean_gpu * 1000, 3),
        "ssd_mean_ms": round(mean_ssd * 1000, 3),
        "total_per_iter_ms": round(total_per_iter * 1000, 3),
        "gpu_gflops": round(flops_per_op / mean_gpu / 1e9, 1),
        "ssd_bandwidth_gbps": round(chunk_bytes / mean_ssd / (1024**3), 3),
        "iterations": iterations,
    }


# ---------------------------------------------------------------------------
# Concurrent pipeline
# ---------------------------------------------------------------------------


def bench_concurrent(
    n: int, filepath: Path, chunk_bytes: int, iterations: int
) -> dict:
    """Benchmark concurrent: GPU compute overlapped with SSD read."""
    a = mx.random.normal((n, n))
    b = mx.random.normal((n, n))
    mx.eval(a, b)

    file_size = filepath.stat().st_size

    gpu_times = []
    ssd_times = []
    wall_times = []

    for i in range(iterations):
        offset = (i * chunk_bytes) % (file_size - chunk_bytes)

        # Shared state for SSD thread
        ssd_result = {"time": 0, "bytes": 0}

        def ssd_read():
            fd = os.open(str(filepath), os.O_RDONLY)
            try:
                t0 = time.perf_counter()
                data = os.pread(fd, chunk_bytes, offset)
                ssd_result["time"] = time.perf_counter() - t0
                ssd_result["bytes"] = len(data)
            finally:
                os.close(fd)

        # Launch SSD read in background thread
        t_wall_start = time.perf_counter()
        ssd_thread = threading.Thread(target=ssd_read)
        ssd_thread.start()

        # GPU compute on main thread
        t_gpu_start = time.perf_counter()
        c = a @ b
        mx.eval(c)
        gpu_time = time.perf_counter() - t_gpu_start

        # Wait for SSD
        ssd_thread.join()
        wall_time = time.perf_counter() - t_wall_start

        gpu_times.append(gpu_time)
        ssd_times.append(ssd_result["time"])
        wall_times.append(wall_time)

    flops_per_op = 2 * n * n * n
    mean_gpu = statistics.mean(gpu_times)
    mean_ssd = statistics.mean(ssd_times)
    mean_wall = statistics.mean(wall_times)

    return {
        "gpu_mean_ms": round(mean_gpu * 1000, 3),
        "ssd_mean_ms": round(mean_ssd * 1000, 3),
        "wall_mean_ms": round(mean_wall * 1000, 3),
        "gpu_gflops": round(flops_per_op / mean_gpu / 1e9, 1),
        "ssd_bandwidth_gbps": round(chunk_bytes / mean_ssd / (1024**3), 3),
        "overlap_efficiency": round(1 - (mean_wall / (mean_gpu + mean_ssd)), 4),
        "iterations": iterations,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Phase 3: GPU/SSD contention test")
    parser.add_argument("--matrix-size", type=int, default=DEFAULT_MATRIX_SIZE)
    parser.add_argument("--file-size-gb", type=float, default=DEFAULT_FILE_SIZE_GB)
    args = parser.parse_args()

    env = get_environment_info()
    chunk_bytes = CHUNK_SIZE_MB * 1024 * 1024
    n = args.matrix_size

    print(f"Phase 3: GPU/SSD Contention Test")
    print(f"Hardware: {env['chip']}, {env['memory_gb']} GB")
    print(f"Matrix: {n}x{n}, Chunk: {CHUNK_SIZE_MB}MB, Iterations: {GPU_ITERATIONS}")
    print(f"Available: {get_available_memory_gb():.1f} GB")

    # Create test file
    tmpdir = tempfile.mkdtemp(prefix="contention_bench_")
    test_file = Path(tmpdir) / "test_data.bin"
    print(f"\nCreating {args.file_size_gb}GB test file...")
    chunk = os.urandom(64 * 1024 * 1024)
    size_bytes = int(args.file_size_gb * 1024**3)
    written = 0
    with open(test_file, "wb") as f:
        while written < size_bytes:
            to_write = min(len(chunk), size_bytes - written)
            f.write(chunk[:to_write])
            written += to_write
    os.sync()

    all_results = {}

    # 1. GPU-only baseline
    print(f"\n--- GPU-only baseline ({n}x{n} matmul) ---")
    for rep in range(REPETITIONS):
        result = bench_gpu_only(n, GPU_ITERATIONS)
        all_results[f"gpu_only_rep{rep}"] = result
        print(f"  Rep {rep+1}: {result['gflops']:.1f} GFLOPS ({result['mean_time_ms']:.1f} ms/op)")

    gpu_baseline_gflops = statistics.mean(
        [all_results[f"gpu_only_rep{i}"]["gflops"] for i in range(REPETITIONS)]
    )

    # 2. SSD-only baseline
    print(f"\n--- SSD-only baseline ({CHUNK_SIZE_MB}MB reads) ---")
    for rep in range(REPETITIONS):
        result = bench_ssd_only(test_file, chunk_bytes, SSD_ITERATIONS)
        all_results[f"ssd_only_rep{rep}"] = result
        print(f"  Rep {rep+1}: {result['bandwidth_gbps']:.2f} GB/s ({result['mean_time_ms']:.2f} ms/read)")

    ssd_baseline_gbps = statistics.mean(
        [all_results[f"ssd_only_rep{i}"]["bandwidth_gbps"] for i in range(REPETITIONS)]
    )

    # 3. Serial pipeline
    print(f"\n--- Serial pipeline (read then compute) ---")
    for rep in range(REPETITIONS):
        result = bench_serial(n, test_file, chunk_bytes, GPU_ITERATIONS)
        all_results[f"serial_rep{rep}"] = result
        print(f"  Rep {rep+1}: GPU {result['gpu_gflops']:.1f} GFLOPS, SSD {result['ssd_bandwidth_gbps']:.2f} GB/s, total {result['total_per_iter_ms']:.1f} ms/iter")

    serial_gpu_gflops = statistics.mean(
        [all_results[f"serial_rep{i}"]["gpu_gflops"] for i in range(REPETITIONS)]
    )

    # 4. Concurrent pipeline
    print(f"\n--- Concurrent pipeline (read + compute overlapped) ---")
    for rep in range(REPETITIONS):
        result = bench_concurrent(n, test_file, chunk_bytes, GPU_ITERATIONS)
        all_results[f"concurrent_rep{rep}"] = result
        print(
            f"  Rep {rep+1}: GPU {result['gpu_gflops']:.1f} GFLOPS, SSD {result['ssd_bandwidth_gbps']:.2f} GB/s, "
            f"wall {result['wall_mean_ms']:.1f} ms, overlap {result['overlap_efficiency']:.1%}"
        )

    concurrent_gpu_gflops = statistics.mean(
        [all_results[f"concurrent_rep{i}"]["gpu_gflops"] for i in range(REPETITIONS)]
    )

    # Summary
    gpu_degradation = 1 - (concurrent_gpu_gflops / gpu_baseline_gflops) if gpu_baseline_gflops > 0 else 0
    serial_degradation = 1 - (serial_gpu_gflops / gpu_baseline_gflops) if gpu_baseline_gflops > 0 else 0

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"GPU baseline:     {gpu_baseline_gflops:.1f} GFLOPS")
    print(f"SSD baseline:     {ssd_baseline_gbps:.2f} GB/s")
    print(f"Serial GPU:       {serial_gpu_gflops:.1f} GFLOPS ({serial_degradation:.1%} degradation)")
    print(f"Concurrent GPU:   {concurrent_gpu_gflops:.1f} GFLOPS ({gpu_degradation:.1%} degradation)")
    print(f"\nVerdict: {'SERIAL PREFERRED' if gpu_degradation > 0.1 else 'CONCURRENT VIABLE'}")
    print(f"  Concurrent GPU degradation: {gpu_degradation:.1%}")
    print(f"  (Flash MOE reported 73% on M2 Ultra)")

    # Cleanup
    test_file.unlink()
    os.rmdir(tmpdir)

    # Log
    log_experiment(
        experiment_name="gpu_ssd_contention",
        phase="gpu_ssd_contention",
        config={
            "matrix_size": n,
            "chunk_size_mb": CHUNK_SIZE_MB,
            "file_size_gb": args.file_size_gb,
            "gpu_iterations": GPU_ITERATIONS,
            "ssd_iterations": SSD_ITERATIONS,
            "repetitions": REPETITIONS,
        },
        results={
            "benchmarks": all_results,
            "summary": {
                "gpu_baseline_gflops": round(gpu_baseline_gflops, 1),
                "ssd_baseline_gbps": round(ssd_baseline_gbps, 3),
                "serial_gpu_gflops": round(serial_gpu_gflops, 1),
                "concurrent_gpu_gflops": round(concurrent_gpu_gflops, 1),
                "concurrent_gpu_degradation": round(gpu_degradation, 4),
                "serial_gpu_degradation": round(serial_degradation, 4),
                "verdict": "serial" if gpu_degradation > 0.1 else "concurrent",
            },
        },
        env=env,
    )


if __name__ == "__main__":
    main()
