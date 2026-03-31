"""
page_cache_bench.py — Phase 2: Page cache behavior under memory pressure.

Simulates MoE expert access patterns with Zipf-distributed popularity and
measures cache residency via mincore(), latency, and pageout activity.

Usage:
    uv run python scripts/page_cache_bench.py [--corpus-size-gb SIZE] [--pressure AVAILABLE_GB]
"""

import argparse
import ctypes
import ctypes.util
import mmap
import os
import random
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

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

DEFAULT_CORPUS_SIZE_GB = 20  # Large enough to exceed usable cache
N_EXPERTS_PER_LAYER = 64
N_LAYERS = 48
K_ACTIVE = 4  # Active experts per token per layer
N_TOKENS_SIMULATED = 200  # Simulated decode steps
ZIPF_EXPONENT = 1.1  # Expert popularity skew (higher = more skewed)


# ---------------------------------------------------------------------------
# mincore via ctypes
# ---------------------------------------------------------------------------

_libc = ctypes.CDLL(ctypes.util.find_library("c"))


def mincore_residency(mm: mmap.mmap, file_size: int) -> float:
    """Check fraction of mmap'd pages resident in memory via mincore().

    Returns fraction in [0, 1].
    """
    page_size = os.sysconf("SC_PAGE_SIZE")
    n_pages = (file_size + page_size - 1) // page_size

    # Get the mmap buffer address
    buf = (ctypes.c_char * file_size).from_buffer(mm)
    addr = ctypes.addressof(buf)

    # mincore output: one byte per page, LSB = 1 if resident
    vec = (ctypes.c_char * n_pages)()
    ret = _libc.mincore(
        ctypes.c_void_p(addr),
        ctypes.c_size_t(file_size),
        ctypes.cast(vec, ctypes.POINTER(ctypes.c_char)),
    )
    if ret != 0:
        return -1.0  # error

    resident = sum(1 for i in range(n_pages) if vec[i] != b"\x00")
    return resident / n_pages


def mincore_region_residency(mm: mmap.mmap, offset: int, length: int) -> float:
    """Check residency for a specific region of the mmap."""
    page_size = os.sysconf("SC_PAGE_SIZE")

    # Align offset down to page boundary
    aligned_offset = (offset // page_size) * page_size
    aligned_length = length + (offset - aligned_offset)
    aligned_length = ((aligned_length + page_size - 1) // page_size) * page_size
    n_pages = aligned_length // page_size

    buf = (ctypes.c_char * len(mm)).from_buffer(mm)
    addr = ctypes.addressof(buf) + aligned_offset

    vec = (ctypes.c_char * n_pages)()
    ret = _libc.mincore(
        ctypes.c_void_p(addr),
        ctypes.c_size_t(aligned_length),
        ctypes.cast(vec, ctypes.POINTER(ctypes.c_char)),
    )
    if ret != 0:
        return -1.0

    resident = sum(1 for i in range(n_pages) if vec[i] != b"\x00")
    return resident / n_pages


# ---------------------------------------------------------------------------
# Expert access pattern simulation
# ---------------------------------------------------------------------------


def generate_zipf_expert_sequence(
    n_experts: int, n_layers: int, k_active: int, n_tokens: int, exponent: float
) -> list[list[tuple[int, int]]]:
    """Generate a sequence of expert activations using Zipf distribution.

    Returns list of token steps, each containing (layer_idx, expert_idx) pairs.
    """
    rng = np.random.default_rng(42)

    # Pre-compute per-layer expert popularity (Zipf)
    # Each layer has its own popularity distribution
    layer_dists = []
    for _ in range(n_layers):
        # Zipf weights: 1/rank^exponent
        ranks = np.arange(1, n_experts + 1, dtype=np.float64)
        weights = 1.0 / np.power(ranks, exponent)
        # Shuffle to avoid all layers favoring the same experts
        perm = rng.permutation(n_experts)
        weights = weights[perm]
        weights /= weights.sum()
        layer_dists.append(weights)

    sequence = []
    for _ in range(n_tokens):
        token_experts = []
        for layer_idx in range(n_layers):
            # Sample k_active experts without replacement
            chosen = rng.choice(
                n_experts, size=k_active, replace=False, p=layer_dists[layer_idx]
            )
            for expert_idx in chosen:
                token_experts.append((layer_idx, int(expert_idx)))
        sequence.append(token_experts)

    return sequence


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------


def create_expert_corpus(size_gb: float, path: Path) -> Path:
    """Create a large synthetic expert corpus file."""
    size_bytes = int(size_gb * 1024**3)
    print(f"  Creating {size_gb}GB expert corpus at {path}...")
    chunk = os.urandom(64 * 1024 * 1024)
    written = 0
    with open(path, "wb") as f:
        while written < size_bytes:
            to_write = min(len(chunk), size_bytes - written)
            f.write(chunk[:to_write])
            written += to_write
    os.sync()
    print(f"  Expert corpus created: {size_bytes / (1024**3):.2f} GB")
    return path


def compute_expert_offset(
    layer_idx: int, expert_idx: int, n_experts: int, expert_size_bytes: int
) -> int:
    """Compute byte offset for an expert in the corpus file."""
    global_idx = layer_idx * n_experts + expert_idx
    return global_idx * expert_size_bytes


def run_cache_benchmark(
    corpus_path: Path,
    corpus_size: int,
    expert_sequence: list[list[tuple[int, int]]],
    n_experts: int,
    expert_size_bytes: int,
    access_method: str,
    cache_policy: str,
) -> dict:
    """Run a single cache benchmark with given access method and policy.

    access_method: 'pread' or 'mmap'
    cache_policy: 'default', 'MADV_SEQUENTIAL', 'MADV_RANDOM', 'F_NOCACHE'
    """
    total_accesses = 0
    total_bytes = 0
    latencies = []
    residency_samples = []

    if access_method == "mmap":
        fd = os.open(str(corpus_path), os.O_RDONLY)
        mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)

        if cache_policy == "MADV_SEQUENTIAL":
            mm.madvise(mmap.MADV_SEQUENTIAL)
        elif cache_policy == "MADV_RANDOM":
            mm.madvise(mmap.MADV_RANDOM)

        try:
            for step, token_experts in enumerate(expert_sequence):
                for layer_idx, expert_idx in token_experts:
                    offset = compute_expert_offset(
                        layer_idx, expert_idx, n_experts, expert_size_bytes
                    )
                    if offset + expert_size_bytes > corpus_size:
                        offset = offset % (corpus_size - expert_size_bytes)

                    # Check residency before access
                    res = mincore_region_residency(mm, offset, expert_size_bytes)
                    if res >= 0:
                        residency_samples.append(res)

                    # Timed access
                    t0 = time.perf_counter()
                    _ = mm[offset : offset + expert_size_bytes]
                    elapsed = time.perf_counter() - t0
                    latencies.append(elapsed * 1000)  # ms
                    total_accesses += 1
                    total_bytes += expert_size_bytes

                # Sample overall residency every 10 steps
                if step % 10 == 0:
                    overall_res = mincore_residency(mm, corpus_size)
                    if overall_res >= 0:
                        pass  # tracked via per-expert samples
        finally:
            mm.close()
            os.close(fd)

    elif access_method == "pread":
        import fcntl

        fd = os.open(str(corpus_path), os.O_RDONLY)
        if cache_policy == "F_NOCACHE":
            fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)

        try:
            for token_experts in expert_sequence:
                for layer_idx, expert_idx in token_experts:
                    offset = compute_expert_offset(
                        layer_idx, expert_idx, n_experts, expert_size_bytes
                    )
                    if offset + expert_size_bytes > corpus_size:
                        offset = offset % (corpus_size - expert_size_bytes)

                    t0 = time.perf_counter()
                    data = os.pread(fd, expert_size_bytes, offset)
                    elapsed = time.perf_counter() - t0
                    latencies.append(elapsed * 1000)
                    total_accesses += 1
                    total_bytes += len(data)
        finally:
            os.close(fd)

    # Compute stats
    latency_sorted = sorted(latencies)
    n = len(latency_sorted)
    result = {
        "access_method": access_method,
        "cache_policy": cache_policy,
        "total_accesses": total_accesses,
        "total_bytes_gb": round(total_bytes / (1024**3), 3),
        "latency_p50_ms": round(latency_sorted[n // 2], 4) if n > 0 else 0,
        "latency_p95_ms": round(latency_sorted[int(n * 0.95)], 4) if n > 0 else 0,
        "latency_mean_ms": round(statistics.mean(latencies), 4) if n > 0 else 0,
        "bandwidth_gbps": round(
            total_bytes / sum(latencies) * 1000 / (1024**3), 3
        )
        if sum(latencies) > 0
        else 0,
    }

    if residency_samples:
        result["cache_residency_mean"] = round(statistics.mean(residency_samples), 4)
        result["cache_residency_min"] = round(min(residency_samples), 4)
    else:
        result["cache_residency_mean"] = None
        result["cache_residency_min"] = None

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Page cache behavior")
    parser.add_argument(
        "--corpus-size-gb",
        type=float,
        default=DEFAULT_CORPUS_SIZE_GB,
        help=f"Expert corpus size in GB (default: {DEFAULT_CORPUS_SIZE_GB})",
    )
    parser.add_argument(
        "--pressure",
        type=float,
        default=None,
        help="Target available memory in GB (creates pressure ballast)",
    )
    parser.add_argument(
        "--expert-size-mb",
        type=float,
        default=4.0,
        help="Size of each expert chunk in MB (default: 4.0)",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=N_TOKENS_SIMULATED,
        help=f"Number of decode steps to simulate (default: {N_TOKENS_SIMULATED})",
    )
    args = parser.parse_args()

    env = get_environment_info()
    print(f"Phase 2: Page Cache Behavior")
    print(f"Hardware: {env['chip']}, {env['memory_gb']} GB")
    print(f"Available: {env['available_gb']:.1f} GB")

    expert_size_bytes = int(args.expert_size_mb * 1024 * 1024)

    # Memory pressure
    cpu_ballast = None
    mlx_ballast = None
    if args.pressure is not None:
        print(f"\nCreating memory pressure (target: {args.pressure} GB available)...")
        cpu_ballast, mlx_ballast = create_memory_pressure(args.pressure)
        actual = get_available_memory_gb()
        print(f"Verified available: {actual:.1f} GB")

    # Create corpus
    tmpdir = tempfile.mkdtemp(prefix="cache_bench_")
    corpus_path = Path(tmpdir) / "expert_corpus.bin"
    create_expert_corpus(args.corpus_size_gb, corpus_path)
    corpus_size = corpus_path.stat().st_size

    # Generate expert access sequence
    print(f"\nGenerating Zipf expert sequence (exponent={ZIPF_EXPONENT})...")
    print(f"  {N_LAYERS} layers x {N_EXPERTS_PER_LAYER} experts, K={K_ACTIVE}, {args.tokens} tokens")
    sequence = generate_zipf_expert_sequence(
        N_EXPERTS_PER_LAYER, N_LAYERS, K_ACTIVE, args.tokens, ZIPF_EXPONENT
    )
    total_expert_accesses = sum(len(step) for step in sequence)
    print(f"  Total expert accesses: {total_expert_accesses}")

    # Run benchmarks across access path x cache policy
    test_configs = [
        ("pread", "default"),
        ("pread", "F_NOCACHE"),
        ("mmap", "default"),
        ("mmap", "MADV_SEQUENTIAL"),
        ("mmap", "MADV_RANDOM"),
    ]

    all_results = {}
    for access_method, cache_policy in test_configs:
        label = f"{access_method}_{cache_policy}"
        print(f"\n--- {access_method} + {cache_policy} ---")

        vm_before = get_vm_stat()
        result = run_cache_benchmark(
            corpus_path,
            corpus_size,
            sequence,
            N_EXPERTS_PER_LAYER,
            expert_size_bytes,
            access_method,
            cache_policy,
        )
        vm_after = get_vm_stat()
        result["vm_stat_delta"] = vm_stat_delta(vm_before, vm_after)

        all_results[label] = result
        print(f"  Latency p50: {result['latency_p50_ms']:.3f} ms, p95: {result['latency_p95_ms']:.3f} ms")
        print(f"  Bandwidth: {result['bandwidth_gbps']:.2f} GB/s")
        if result["cache_residency_mean"] is not None:
            print(f"  Cache residency: {result['cache_residency_mean']:.1%} mean")
        print(f"  Pageout delta: {result['vm_stat_delta']['pageout_delta_mb']:.1f} MB")

    # Summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<25} {'p50 ms':>8} {'p95 ms':>8} {'GB/s':>8} {'Residency':>10} {'Pageout MB':>10}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
    for label, r in all_results.items():
        res = f"{r['cache_residency_mean']:.1%}" if r["cache_residency_mean"] is not None else "N/A"
        print(
            f"{label:<25} {r['latency_p50_ms']:>8.3f} {r['latency_p95_ms']:>8.3f} "
            f"{r['bandwidth_gbps']:>8.2f} {res:>10} {r['vm_stat_delta']['pageout_delta_mb']:>10.1f}"
        )

    # Cleanup
    corpus_path.unlink()
    os.rmdir(tmpdir)

    # Log
    log_experiment(
        experiment_name=f"page_cache_{'pressure_' + str(int(args.pressure)) + 'gb' if args.pressure else 'no_pressure'}",
        phase="page_cache",
        config={
            "corpus_size_gb": args.corpus_size_gb,
            "expert_size_mb": args.expert_size_mb,
            "n_experts_per_layer": N_EXPERTS_PER_LAYER,
            "n_layers": N_LAYERS,
            "k_active": K_ACTIVE,
            "n_tokens": args.tokens,
            "zipf_exponent": ZIPF_EXPONENT,
            "pressure_target_gb": args.pressure,
            "actual_available_gb": round(get_available_memory_gb(), 2),
        },
        results=all_results,
        env=env,
    )

    del cpu_ballast, mlx_ballast


if __name__ == "__main__":
    main()
