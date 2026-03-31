"""
nvme_profile.py — Phase 1: Profile M4 NVMe read performance.

Tests sequential pread(), mmap, and scattered reads at various chunk sizes.
Runs under configurable memory pressure to simulate real inference conditions.

Usage:
    uv run python scripts/nvme_profile.py [--pressure AVAILABLE_GB] [--file-size-gb SIZE]
"""

import argparse
import mmap
import os
import statistics
import tempfile
import time
from pathlib import Path

from experiment_utils import (
    create_memory_pressure,
    get_available_memory_gb,
    get_environment_info,
    get_vm_stat,
    log_experiment,
    vm_stat_delta,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHUNK_SIZES_MB = [1, 2, 4, 8]
REPETITIONS = 5
DEFAULT_FILE_SIZE_GB = 2  # Test file size


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------


def create_test_file(size_gb: float, path: Path) -> Path:
    """Create a test file with random-ish data for benchmarking."""
    size_bytes = int(size_gb * 1024**3)
    print(f"  Creating {size_gb}GB test file at {path}...")

    # Write in 64MB chunks
    chunk = os.urandom(64 * 1024 * 1024)
    written = 0
    with open(path, "wb") as f:
        while written < size_bytes:
            to_write = min(len(chunk), size_bytes - written)
            f.write(chunk[:to_write])
            written += to_write

    # Sync to disk
    os.sync()
    print(f"  Test file created: {size_bytes / (1024**3):.2f} GB")
    return path


def purge_cache():
    """Attempt to purge the OS page cache. Requires sudo."""
    try:
        os.system("sync")
        ret = os.system("sudo purge 2>/dev/null")
        if ret != 0:
            print("  WARNING: 'sudo purge' failed — cache may be warm")
            return False
        return True
    except Exception:
        return False


def bench_pread(filepath: Path, chunk_size_bytes: int, file_size: int, reps: int) -> list[dict]:
    """Benchmark pread() at given chunk size."""
    results = []
    fd = os.open(str(filepath), os.O_RDONLY)
    try:
        for _ in range(reps):
            offset = 0
            bytes_read = 0
            t0 = time.perf_counter()
            while offset < file_size:
                n = os.pread(fd, chunk_size_bytes, offset)
                bytes_read += len(n)
                offset += chunk_size_bytes
            elapsed = time.perf_counter() - t0
            bandwidth = bytes_read / elapsed / (1024**3)
            latency_ms = elapsed / (file_size / chunk_size_bytes) * 1000
            results.append({
                "bandwidth_gbps": round(bandwidth, 3),
                "latency_ms_per_chunk": round(latency_ms, 3),
                "elapsed_s": round(elapsed, 4),
                "bytes_read": bytes_read,
            })
    finally:
        os.close(fd)
    return results


def bench_pread_nocache(filepath: Path, chunk_size_bytes: int, file_size: int, reps: int) -> list[dict]:
    """Benchmark pread() with F_NOCACHE (bypass page cache)."""
    import fcntl

    results = []
    fd = os.open(str(filepath), os.O_RDONLY)
    try:
        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        for _ in range(reps):
            offset = 0
            bytes_read = 0
            t0 = time.perf_counter()
            while offset < file_size:
                n = os.pread(fd, chunk_size_bytes, offset)
                bytes_read += len(n)
                offset += chunk_size_bytes
            elapsed = time.perf_counter() - t0
            bandwidth = bytes_read / elapsed / (1024**3)
            latency_ms = elapsed / (file_size / chunk_size_bytes) * 1000
            results.append({
                "bandwidth_gbps": round(bandwidth, 3),
                "latency_ms_per_chunk": round(latency_ms, 3),
                "elapsed_s": round(elapsed, 4),
                "bytes_read": bytes_read,
            })
    finally:
        os.close(fd)
    return results


def bench_mmap_sequential(filepath: Path, chunk_size_bytes: int, file_size: int, reps: int) -> list[dict]:
    """Benchmark mmap with MADV_SEQUENTIAL."""
    results = []
    for _ in range(reps):
        fd = os.open(str(filepath), os.O_RDONLY)
        try:
            mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            mm.madvise(mmap.MADV_SEQUENTIAL)
            offset = 0
            bytes_read = 0
            t0 = time.perf_counter()
            while offset < file_size:
                end = min(offset + chunk_size_bytes, file_size)
                _ = mm[offset:end]
                bytes_read += end - offset
                offset = end
            elapsed = time.perf_counter() - t0
            bandwidth = bytes_read / elapsed / (1024**3)
            latency_ms = elapsed / (file_size / chunk_size_bytes) * 1000
            results.append({
                "bandwidth_gbps": round(bandwidth, 3),
                "latency_ms_per_chunk": round(latency_ms, 3),
                "elapsed_s": round(elapsed, 4),
                "bytes_read": bytes_read,
            })
            mm.close()
        finally:
            os.close(fd)
    return results


def bench_mmap_random(filepath: Path, chunk_size_bytes: int, file_size: int, reps: int) -> list[dict]:
    """Benchmark mmap with MADV_RANDOM (simulates scattered expert access)."""
    import random

    results = []
    n_chunks = file_size // chunk_size_bytes
    for _ in range(reps):
        fd = os.open(str(filepath), os.O_RDONLY)
        try:
            mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            mm.madvise(mmap.MADV_RANDOM)
            # Random access pattern
            offsets = [random.randint(0, n_chunks - 1) * chunk_size_bytes for _ in range(n_chunks)]
            bytes_read = 0
            t0 = time.perf_counter()
            for offset in offsets:
                end = min(offset + chunk_size_bytes, file_size)
                _ = mm[offset:end]
                bytes_read += end - offset
            elapsed = time.perf_counter() - t0
            bandwidth = bytes_read / elapsed / (1024**3)
            latency_ms = elapsed / len(offsets) * 1000
            results.append({
                "bandwidth_gbps": round(bandwidth, 3),
                "latency_ms_per_chunk": round(latency_ms, 3),
                "elapsed_s": round(elapsed, 4),
                "bytes_read": bytes_read,
            })
            mm.close()
        finally:
            os.close(fd)
    return results


def bench_scattered_reads(filepath: Path, chunk_size_bytes: int, file_size: int, reps: int, scatter_count: int = 3) -> list[dict]:
    """Benchmark scattered reads simulating expert tensors across shard boundaries.

    Reads scatter_count non-contiguous chunks per 'expert', simulating
    gate_proj + up_proj + down_proj stored in different locations.
    """
    import random

    results = []
    n_chunks = file_size // chunk_size_bytes
    sub_chunk = chunk_size_bytes // scatter_count

    for _ in range(reps):
        fd = os.open(str(filepath), os.O_RDONLY)
        try:
            # Generate scattered offset patterns
            n_experts = n_chunks // scatter_count
            bytes_read = 0
            t0 = time.perf_counter()
            for _ in range(n_experts):
                # Read scatter_count non-contiguous sub-chunks
                for _ in range(scatter_count):
                    offset = random.randint(0, file_size - sub_chunk)
                    data = os.pread(fd, sub_chunk, offset)
                    bytes_read += len(data)
            elapsed = time.perf_counter() - t0
            bandwidth = bytes_read / elapsed / (1024**3)
            latency_ms = elapsed / n_experts * 1000
            results.append({
                "bandwidth_gbps": round(bandwidth, 3),
                "latency_ms_per_expert": round(latency_ms, 3),
                "elapsed_s": round(elapsed, 4),
                "bytes_read": bytes_read,
                "experts_loaded": n_experts,
            })
        finally:
            os.close(fd)
    return results


def summarize_results(runs: list[dict], key: str = "bandwidth_gbps") -> dict:
    """Compute summary stats from a list of benchmark runs."""
    values = [r[key] for r in runs]
    return {
        f"{key}_mean": round(statistics.mean(values), 3),
        f"{key}_std": round(statistics.stdev(values), 3) if len(values) > 1 else 0,
        f"{key}_min": round(min(values), 3),
        f"{key}_max": round(max(values), 3),
        "runs": len(runs),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Phase 1: NVMe read profiling")
    parser.add_argument(
        "--pressure",
        type=float,
        default=None,
        help="Target available memory in GB (creates pressure ballast)",
    )
    parser.add_argument(
        "--file-size-gb",
        type=float,
        default=DEFAULT_FILE_SIZE_GB,
        help=f"Test file size in GB (default: {DEFAULT_FILE_SIZE_GB})",
    )
    args = parser.parse_args()

    env = get_environment_info()
    print(f"Phase 1: NVMe Profiling")
    print(f"Hardware: {env['chip']}, {env['memory_gb']} GB")
    print(f"Available: {env['available_gb']:.1f} GB")

    # Memory pressure
    cpu_ballast = None
    mlx_ballast = None
    if args.pressure is not None:
        print(f"\nCreating memory pressure (target: {args.pressure} GB available)...")
        cpu_ballast, mlx_ballast = create_memory_pressure(args.pressure)
        actual = get_available_memory_gb()
        print(f"Verified available: {actual:.1f} GB")

    # Create test file
    tmpdir = tempfile.mkdtemp(prefix="nvme_bench_")
    test_file = Path(tmpdir) / "test_data.bin"
    create_test_file(args.file_size_gb, test_file)
    file_size = test_file.stat().st_size

    all_results = {}

    for chunk_mb in CHUNK_SIZES_MB:
        chunk_bytes = chunk_mb * 1024 * 1024
        print(f"\n--- Chunk size: {chunk_mb} MB ---")

        # 1. pread (warm cache)
        print(f"  pread (warm cache)...")
        runs = bench_pread(test_file, chunk_bytes, file_size, REPETITIONS)
        summary = summarize_results(runs)
        all_results[f"pread_{chunk_mb}mb_warm"] = {"runs": runs, "summary": summary}
        print(f"    {summary['bandwidth_gbps_mean']:.2f} GB/s (std: {summary['bandwidth_gbps_std']:.2f})")

        # 2. pread + F_NOCACHE (cold / bypass cache)
        print(f"  pread + F_NOCACHE...")
        runs = bench_pread_nocache(test_file, chunk_bytes, file_size, REPETITIONS)
        summary = summarize_results(runs)
        all_results[f"pread_{chunk_mb}mb_nocache"] = {"runs": runs, "summary": summary}
        print(f"    {summary['bandwidth_gbps_mean']:.2f} GB/s (std: {summary['bandwidth_gbps_std']:.2f})")

        # 3. mmap + MADV_SEQUENTIAL (warm)
        print(f"  mmap + MADV_SEQUENTIAL...")
        runs = bench_mmap_sequential(test_file, chunk_bytes, file_size, REPETITIONS)
        summary = summarize_results(runs)
        all_results[f"mmap_seq_{chunk_mb}mb"] = {"runs": runs, "summary": summary}
        print(f"    {summary['bandwidth_gbps_mean']:.2f} GB/s (std: {summary['bandwidth_gbps_std']:.2f})")

        # 4. mmap + MADV_RANDOM
        print(f"  mmap + MADV_RANDOM...")
        runs = bench_mmap_random(test_file, chunk_bytes, file_size, REPETITIONS)
        summary = summarize_results(runs)
        all_results[f"mmap_random_{chunk_mb}mb"] = {"runs": runs, "summary": summary}
        print(f"    {summary['bandwidth_gbps_mean']:.2f} GB/s (std: {summary['bandwidth_gbps_std']:.2f})")

        # 5. Scattered reads (simulating non-contiguous expert layout)
        print(f"  Scattered reads (3 sub-chunks per expert)...")
        runs = bench_scattered_reads(test_file, chunk_bytes, file_size, REPETITIONS)
        summary = summarize_results(runs)
        all_results[f"scattered_{chunk_mb}mb"] = {"runs": runs, "summary": summary}
        print(f"    {summary['bandwidth_gbps_mean']:.2f} GB/s (std: {summary['bandwidth_gbps_std']:.2f})")

    # Cleanup test file
    test_file.unlink()
    os.rmdir(tmpdir)

    # Summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY (mean GB/s)")
    print(f"{'='*60}")
    print(f"{'Method':<30} {'1MB':>8} {'2MB':>8} {'4MB':>8} {'8MB':>8}")
    print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for method in ["pread", "pread_nocache", "mmap_seq", "mmap_random", "scattered"]:
        label = method.replace("_", " ")
        vals = []
        for chunk_mb in CHUNK_SIZES_MB:
            key = f"{method}_{chunk_mb}mb"
            if key.startswith("pread_") and "nocache" not in key:
                key = f"pread_{chunk_mb}mb_warm"
            s = all_results.get(key, {}).get("summary", {})
            vals.append(f"{s.get('bandwidth_gbps_mean', 0):>7.2f}")
        print(f"{label:<30} {'  '.join(vals)}")

    # Log results
    log_experiment(
        experiment_name=f"nvme_profile_{'pressure_' + str(int(args.pressure)) + 'gb' if args.pressure else 'no_pressure'}",
        phase="nvme_profile",
        config={
            "chunk_sizes_mb": CHUNK_SIZES_MB,
            "repetitions": REPETITIONS,
            "file_size_gb": args.file_size_gb,
            "pressure_target_gb": args.pressure,
            "actual_available_gb": round(get_available_memory_gb(), 2),
        },
        results=all_results,
        env=env,
    )

    # Release ballast
    del cpu_ballast, mlx_ballast


if __name__ == "__main__":
    main()
