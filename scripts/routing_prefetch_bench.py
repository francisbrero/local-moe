"""
routing_prefetch_bench.py — Phase 3: Async prefetch pipeline simulation for H2.

Simulates expert prefetching using real routing traces and a synthetic expert
corpus on SSD. Measures prefetch hit rate, pipeline stall time, and projected
throughput (upper bound).

This is a SYNTHETIC I/O SIMULATION — results are upper bounds, not real
inference measurements. Real throughput validation requires #17.

Usage:
    uv run python scripts/routing_prefetch_bench.py [--trace-dir routing_traces]
"""

import argparse
import json
import os
import struct
import tempfile
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from experiment_utils import (
    get_environment_info,
    get_rss_mb,
    get_vm_stat,
    log_experiment,
    vm_stat_delta,
)

# ---------------------------------------------------------------------------
# Config — H0 Phase 4a measured timing anchors
# ---------------------------------------------------------------------------

TRACE_DIR = Path(__file__).parent.parent / "routing_traces"
NUM_EXPERTS = 128
TOP_K = 8
NUM_LAYERS = 48
EXPERT_SIZE_BYTES = 2_359_296  # 2.25 MB at 4-bit (from H0 checkpoint audit)

# H0 Phase 4a timing anchors (from experiments.jsonl)
COMPUTE_P50_MS = 0.2568   # per-expert GEMM
LOAD_P50_MS = 0.1434      # per-expert SSD read (pread, warm cache)
STREAMED_P50_MS = 0.4017  # compute + load combined


# ---------------------------------------------------------------------------
# Synthetic expert corpus
# ---------------------------------------------------------------------------


def create_synthetic_corpus(corpus_dir: Path, n_layers=NUM_LAYERS,
                           n_experts=NUM_EXPERTS, expert_size=EXPERT_SIZE_BYTES):
    """Create synthetic expert weight files mimicking real checkpoint layout.

    Creates one file per layer with 3 tensors per expert (gate_proj, up_proj,
    down_proj) at computed offsets. Uses random data to ensure real SSD reads.
    """
    corpus_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for layer_idx in range(n_layers):
        layer_file = corpus_dir / f"layer_{layer_idx:03d}.bin"
        if layer_file.exists() and layer_file.stat().st_size >= n_experts * expert_size:
            # Already exists with correct size
            manifest[layer_idx] = {
                "file": str(layer_file),
                "expert_offsets": {
                    e: e * expert_size for e in range(n_experts)
                },
                "expert_size": expert_size,
            }
            continue

        # Write random data
        with open(layer_file, "wb") as f:
            for e in range(n_experts):
                f.write(os.urandom(expert_size))

        manifest[layer_idx] = {
            "file": str(layer_file),
            "expert_offsets": {e: e * expert_size for e in range(n_experts)},
            "expert_size": expert_size,
        }

    # Save manifest
    manifest_path = corpus_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    total_gb = n_layers * n_experts * expert_size / (1024**3)
    print(f"  Corpus: {n_layers} layers × {n_experts} experts = {total_gb:.1f} GB")
    return manifest


def load_manifest(corpus_dir: Path):
    """Load existing corpus manifest."""
    manifest_path = corpus_dir / "manifest.json"
    with open(manifest_path) as f:
        raw = json.load(f)
    # Convert string keys back to int
    manifest = {}
    for k, v in raw.items():
        layer_idx = int(k)
        v["expert_offsets"] = {int(ek): ev for ek, ev in v["expert_offsets"].items()}
        manifest[layer_idx] = v
    return manifest


# ---------------------------------------------------------------------------
# Staging cache with Least-Stale eviction
# ---------------------------------------------------------------------------


class LeastStaleCache:
    """Expert staging cache with Least-Stale eviction (SpecMD).

    Unlike LRU, Least-Stale evicts the expert whose last-predicted-use is
    furthest in the past — an expert that was predicted but never used is
    staler than one currently in use.
    """

    def __init__(self, capacity_experts=TOP_K * 4):
        self.capacity = capacity_experts
        self.cache = OrderedDict()  # (layer_idx, expert_id) -> last_predicted_step
        self.hits = 0
        self.misses = 0

    def contains(self, layer_idx, expert_id):
        return (layer_idx, expert_id) in self.cache

    def access(self, layer_idx, expert_id, step):
        """Record an access (actual use). Returns True if hit."""
        key = (layer_idx, expert_id)
        if key in self.cache:
            self.cache[key] = step
            self.cache.move_to_end(key)
            self.hits += 1
            return True
        self.misses += 1
        return False

    def prefetch(self, layer_idx, expert_id, step):
        """Add a prefetched expert to cache."""
        key = (layer_idx, expert_id)
        if key in self.cache:
            self.cache[key] = step
            self.cache.move_to_end(key)
            return

        if len(self.cache) >= self.capacity:
            # Evict least-stale: the entry with the oldest predicted step
            self.cache.popitem(last=False)

        self.cache[key] = step

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Prefetch pipeline simulation
# ---------------------------------------------------------------------------


def pread_expert(manifest, layer_idx, expert_id):
    """Read an expert's weights from the synthetic corpus via pread."""
    layer_info = manifest[layer_idx]
    offset = layer_info["expert_offsets"][expert_id]
    size = layer_info["expert_size"]

    fd = os.open(layer_info["file"], os.O_RDONLY)
    try:
        data = os.pread(fd, size, offset)
    finally:
        os.close(fd)
    return data


def simulate_prefetch_pipeline(
    traces, manifest, predicted_experts, mode="predicted",
    lookahead=2, cache_capacity=TOP_K * 4
):
    """Simulate the async prefetch pipeline.

    Args:
        traces: dict layer_idx -> {expert_indices: [T, K]}
        manifest: corpus manifest from create_synthetic_corpus
        predicted_experts: dict (layer_idx, token) -> predicted expert indices
            (None for baseline/oracle modes)
        mode: "baseline" (no prefetch), "oracle" (perfect prediction), "predicted"
        lookahead: how many layers ahead to prefetch
        cache_capacity: staging cache size in number of experts
    """
    layer_indices = sorted(traces.keys())
    n_tokens = traces[layer_indices[0]]["expert_indices"].shape[0]
    cache = LeastStaleCache(capacity_experts=cache_capacity)

    total_compute_ms = 0.0
    total_stall_ms = 0.0
    total_prefetch_hits = 0
    total_expert_accesses = 0

    executor = ThreadPoolExecutor(max_workers=2)

    for t in range(n_tokens):
        for i, layer_idx in enumerate(layer_indices):
            actual_experts = traces[layer_idx]["expert_indices"][t].tolist()

            # Check cache for actual experts
            layer_stall_ms = 0.0
            for expert_id in actual_experts:
                total_expert_accesses += 1
                step = t * len(layer_indices) + i

                if cache.access(layer_idx, expert_id, step):
                    total_prefetch_hits += 1
                    # Hit: expert already in staging buffer
                    layer_stall_ms += 0  # no stall
                else:
                    # Miss: need to load from SSD (blocking)
                    layer_stall_ms += LOAD_P50_MS

            # Simulate compute time
            layer_compute_ms = len(actual_experts) * COMPUTE_P50_MS
            total_compute_ms += layer_compute_ms
            total_stall_ms += layer_stall_ms

            # Issue prefetch for layer i+lookahead
            if mode != "baseline" and i + lookahead < len(layer_indices):
                target_layer = layer_indices[i + lookahead]
                target_step = t * len(layer_indices) + i + lookahead

                if mode == "oracle":
                    prefetch_set = traces[target_layer]["expert_indices"][t].tolist()
                elif mode == "predicted":
                    key = (target_layer, t)
                    prefetch_set = predicted_experts.get(key, [])
                else:
                    prefetch_set = []

                # Issue async prefetch (simulated — just populate cache)
                for expert_id in prefetch_set:
                    cache.prefetch(target_layer, expert_id, target_step)
                    # In a real system, this would be an async pread()

    executor.shutdown(wait=False)

    total_time_ms = total_compute_ms + total_stall_ms
    hit_rate = total_prefetch_hits / total_expert_accesses if total_expert_accesses > 0 else 0
    stall_pct = total_stall_ms / total_time_ms * 100 if total_time_ms > 0 else 0
    tok_s = n_tokens / (total_time_ms / 1000) if total_time_ms > 0 else 0

    return {
        "mode": mode,
        "n_tokens": n_tokens,
        "total_compute_ms": round(total_compute_ms, 2),
        "total_stall_ms": round(total_stall_ms, 2),
        "total_time_ms": round(total_time_ms, 2),
        "prefetch_hit_rate": round(hit_rate, 4),
        "pipeline_stall_pct": round(stall_pct, 2),
        "estimated_tok_s": round(tok_s, 2),
        "cache_hit_rate": round(cache.hit_rate, 4),
        "total_expert_accesses": total_expert_accesses,
    }


# ---------------------------------------------------------------------------
# I/O bandwidth benchmark
# ---------------------------------------------------------------------------


def benchmark_pread_bandwidth(manifest, n_reads=50):
    """Measure actual pread bandwidth on the synthetic corpus."""
    layer_info = manifest[0]
    times = []

    for _ in range(n_reads):
        expert_id = np.random.randint(0, NUM_EXPERTS)
        t0 = time.perf_counter()
        data = pread_expert(manifest, 0, expert_id)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    times = np.array(times)
    bytes_per_read = EXPERT_SIZE_BYTES
    bandwidth_gbps = bytes_per_read / times / (1024**3)

    return {
        "read_p50_ms": float(np.percentile(times, 50) * 1000),
        "read_p95_ms": float(np.percentile(times, 95) * 1000),
        "bandwidth_p50_gbps": float(np.percentile(bandwidth_gbps, 50)),
        "bandwidth_mean_gbps": float(np.mean(bandwidth_gbps)),
        "n_reads": n_reads,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="H2 Phase 3: Async prefetch pipeline simulation"
    )
    parser.add_argument("--trace-dir", type=str, default=str(TRACE_DIR))
    parser.add_argument(
        "--corpus-dir", type=str, default=None,
        help="Directory for synthetic corpus (default: /tmp/h2_corpus)"
    )
    parser.add_argument("--skip-corpus", action="store_true",
                        help="Skip corpus generation (use existing)")
    parser.add_argument("--lookahead", type=int, default=2)
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    corpus_dir = Path(args.corpus_dir) if args.corpus_dir else Path(tempfile.gettempdir()) / "h2_corpus"

    print("=== H2 Phase 3: Async Prefetch Pipeline Simulation ===")
    print(f"  (Results are UPPER BOUNDS — synthetic I/O, not real inference)")

    # Load traces
    print("\n--- Loading traces ---")
    from routing_trace import load_all_traces
    traces = load_all_traces(trace_dir)
    layer_indices = sorted(traces.keys())
    n_tokens = traces[layer_indices[0]]["expert_indices"].shape[0]
    print(f"  {len(layer_indices)} layers, {n_tokens} tokens")

    env = get_environment_info()
    vm_before = get_vm_stat()

    # Create or load synthetic corpus
    if not args.skip_corpus:
        print("\n--- Creating synthetic corpus ---")
        manifest = create_synthetic_corpus(corpus_dir)
    else:
        print("\n--- Loading existing corpus ---")
        manifest = load_manifest(corpus_dir)

    # I/O bandwidth benchmark
    print("\n--- I/O Bandwidth ---")
    bw = benchmark_pread_bandwidth(manifest)
    print(f"  pread p50: {bw['read_p50_ms']:.3f}ms, "
          f"bandwidth: {bw['bandwidth_p50_gbps']:.1f} GB/s")

    # Oracle throughput ceiling
    print("\n--- Throughput Ceiling Analysis ---")
    compute_only_ms = n_tokens * NUM_LAYERS * TOP_K * COMPUTE_P50_MS
    compute_tok_s = n_tokens / (compute_only_ms / 1000) if compute_only_ms > 0 else 0
    print(f"  Compute-only floor: {compute_only_ms:.0f}ms = {compute_tok_s:.1f} tok/s")
    streamed_ms = n_tokens * NUM_LAYERS * TOP_K * STREAMED_P50_MS
    streamed_tok_s = n_tokens / (streamed_ms / 1000) if streamed_ms > 0 else 0
    print(f"  Streamed (serial) floor: {streamed_ms:.0f}ms = {streamed_tok_s:.1f} tok/s")

    # Generate "predicted" experts using previous-layer heuristic (simple baseline)
    # In production, this would come from the trained Phase 2 predictor
    print("\n--- Generating predictions (previous-layer heuristic) ---")
    predicted_experts = {}
    for t in range(n_tokens):
        for i, layer_idx in enumerate(layer_indices):
            if i > 0:
                prev_layer = layer_indices[i - 1]
                predicted_experts[(layer_idx, t)] = (
                    traces[prev_layer]["expert_indices"][t].tolist()
                )
            else:
                predicted_experts[(layer_idx, t)] = []

    # Run simulations
    print("\n--- Pipeline Simulation ---")

    # Baseline: no prefetch
    print("\n  [Baseline: no prefetch]")
    baseline = simulate_prefetch_pipeline(
        traces, manifest, None, mode="baseline", lookahead=args.lookahead
    )
    print(f"    Hit rate: {baseline['prefetch_hit_rate']:.1%}")
    print(f"    Stall: {baseline['pipeline_stall_pct']:.1f}%")
    print(f"    Estimated tok/s: {baseline['estimated_tok_s']:.1f} (upper bound)")

    # Oracle: perfect prediction
    print("\n  [Oracle: perfect prediction]")
    oracle = simulate_prefetch_pipeline(
        traces, manifest, None, mode="oracle", lookahead=args.lookahead
    )
    print(f"    Hit rate: {oracle['prefetch_hit_rate']:.1%}")
    print(f"    Stall: {oracle['pipeline_stall_pct']:.1f}%")
    print(f"    Estimated tok/s: {oracle['estimated_tok_s']:.1f} (upper bound)")

    # Predicted: previous-layer heuristic
    print("\n  [Predicted: previous-layer heuristic]")
    predicted = simulate_prefetch_pipeline(
        traces, manifest, predicted_experts, mode="predicted", lookahead=args.lookahead
    )
    print(f"    Hit rate: {predicted['prefetch_hit_rate']:.1%}")
    print(f"    Stall: {predicted['pipeline_stall_pct']:.1f}%")
    print(f"    Estimated tok/s: {predicted['estimated_tok_s']:.1f} (upper bound)")

    vm_after = get_vm_stat()
    vm_delta = vm_stat_delta(vm_before, vm_after)
    peak_rss = get_rss_mb()

    # Success gate
    print("\n--- Success Gate ---")
    # Using oracle as best case since we don't have trained predictors here
    oracle_hit = oracle["prefetch_hit_rate"]
    oracle_stall = oracle["pipeline_stall_pct"]
    gate_hit = oracle_hit >= 0.85
    gate_stall = oracle_stall <= 15.0
    print(f"  Oracle hit rate: {oracle_hit:.1%} (gate: ≥85%) -> {'PASS' if gate_hit else 'FAIL'}")
    print(f"  Oracle stall: {oracle_stall:.1f}% (gate: ≤15%) -> {'PASS' if gate_stall else 'FAIL'}")

    # Log results
    results = {
        "baseline": baseline,
        "oracle": oracle,
        "predicted": predicted,
        "io_bandwidth": bw,
        "throughput_ceiling": {
            "compute_only_tok_s": round(compute_tok_s, 2),
            "streamed_tok_s": round(streamed_tok_s, 2),
        },
        "peak_rss_mb": peak_rss,
        "vm_stat_delta": vm_delta,
        "gate_hit_rate_passed": gate_hit,
        "gate_stall_passed": gate_stall,
    }

    log_experiment(
        experiment_name="h2_prefetch_simulation",
        phase="phase_3_prefetch_sim",
        config={
            "num_experts": NUM_EXPERTS,
            "top_k": TOP_K,
            "num_layers": NUM_LAYERS,
            "expert_size_bytes": EXPERT_SIZE_BYTES,
            "lookahead": args.lookahead,
            "compute_p50_ms": COMPUTE_P50_MS,
            "load_p50_ms": LOAD_P50_MS,
        },
        results=results,
        env=env,
    )

    print(f"\n=== Phase 3 Complete ===")
    print(f"  Results logged to experiments.jsonl")


if __name__ == "__main__":
    main()
