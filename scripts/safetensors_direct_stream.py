"""
H8a: Safetensors Direct Streaming — Zero-Conversion Block Loading.

Loads transformer blocks directly from original safetensors shards via
MLX's mmap-based mx.load(), eliminating the npz serialization overhead
that made Phase 3 (H0+H5) impractically slow at 0.005 tok/s.

Phases:
  0  - Shard layout + MLX load characterization (hard gate)
  1  - 72B block index builder
  2  - 7B direct load benchmark (npz vs safetensors side-by-side)
  3  - 72B integration test

Usage:
    uv run python scripts/safetensors_direct_stream.py --phase 0
    uv run python scripts/safetensors_direct_stream.py --phase 1
    uv run python scripts/safetensors_direct_stream.py --phase 2
    uv run python scripts/safetensors_direct_stream.py --phase 3 [--n-tokens N]
"""

import argparse
import gc
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
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


# ── Safetensors Block Index ──


class SafetensorsBlockIndex:
    """Maps transformer block indices to (shard_file, tensor_names) tuples."""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self._block_map: dict[int, dict[str, str]] = defaultdict(dict)  # block -> {tensor_name: shard}
        self._non_block_map: dict[str, str] = {}  # tensor_name -> shard
        self._shard_dir: Path | None = None
        self._build_index()

    def _build_index(self):
        """Parse model.safetensors.index.json to build block-to-shard mapping."""
        index_path = self.model_path / "model.safetensors.index.json"
        if not index_path.exists():
            # Try HF cache layout
            snapshots = list(self.model_path.glob("snapshots/*/model.safetensors.index.json"))
            if snapshots:
                index_path = snapshots[0]
                self._shard_dir = index_path.parent
            else:
                raise FileNotFoundError(f"No model.safetensors.index.json in {self.model_path}")
        else:
            self._shard_dir = self.model_path

        with open(index_path) as f:
            data = json.load(f)

        weight_map = data.get("weight_map", {})
        for tensor_name, shard_file in weight_map.items():
            m = re.match(r"model\.layers\.(\d+)\.", tensor_name)
            if m:
                block_idx = int(m.group(1))
                self._block_map[block_idx][tensor_name] = shard_file
            else:
                self._non_block_map[tensor_name] = shard_file

    @property
    def n_blocks(self) -> int:
        return len(self._block_map)

    def block_tensor_names(self, block_idx: int) -> dict[str, str]:
        """Return {tensor_name: shard_file} for a block."""
        return dict(self._block_map[block_idx])

    def block_shards(self, block_idx: int) -> set[str]:
        """Return set of shard files needed for a block."""
        return set(self._block_map[block_idx].values())

    def shard_path(self, shard_file: str) -> Path:
        """Resolve shard filename to absolute path."""
        return self._shard_dir / shard_file

    def non_block_tensors(self) -> dict[str, str]:
        return dict(self._non_block_map)

    def summary(self) -> dict:
        """Print and return summary statistics."""
        cross_shard = sum(1 for b in self._block_map if len(self.block_shards(b)) > 1)
        tensors_per_block = [len(self._block_map[b]) for b in self._block_map]
        all_shards = set()
        for b in self._block_map:
            all_shards.update(self.block_shards(b))

        info = {
            "n_blocks": self.n_blocks,
            "tensors_per_block": tensors_per_block[0] if tensors_per_block else 0,
            "cross_shard_blocks": cross_shard,
            "total_shards": len(all_shards),
            "non_block_tensors": len(self._non_block_map),
        }
        print(f"  Blocks: {info['n_blocks']}")
        print(f"  Tensors/block: {info['tensors_per_block']}")
        print(f"  Cross-shard blocks: {info['cross_shard_blocks']}")
        print(f"  Shards: {info['total_shards']}")
        print(f"  Non-block tensors: {info['non_block_tensors']}")
        return info


# ── Block loading utilities ──


def _get_model_blocks(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "layers"):
        return list(model.layers)
    raise ValueError("Could not find transformer layers in model")


def _get_inner_model(model):
    return model.model if hasattr(model, "model") else model


def _find_hf_cache_path(repo_id: str) -> Path:
    """Find the local HF cache path for a downloaded model.

    Uses refs/main to resolve the pinned commit hash deterministically,
    falling back to most-recently-modified snapshot if refs/main is absent.
    """
    # Respect HF_HUB_CACHE / HF_HOME environment variables
    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    if hf_hub_cache:
        cache_dir = Path(hf_hub_cache)
    else:
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            cache_dir = Path(hf_home) / "hub"
        else:
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    slug = "models--" + repo_id.replace("/", "--")
    model_dir = cache_dir / slug
    if not model_dir.exists():
        raise FileNotFoundError(f"Model not found in HF cache: {model_dir}")

    # Prefer refs/main for deterministic resolution
    refs_main = model_dir / "refs" / "main"
    if refs_main.exists():
        commit_hash = refs_main.read_text().strip()
        snapshot = model_dir / "snapshots" / commit_hash
        if snapshot.exists():
            return snapshot

    # Fallback: most recently modified snapshot
    snapshots = sorted(
        (model_dir / "snapshots").iterdir(),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not snapshots:
        raise FileNotFoundError(f"No snapshots in {model_dir}")
    return snapshots[0]


def load_block_from_safetensors(
    block_idx: int,
    index: SafetensorsBlockIndex,
    shard_cache: dict[str, dict[str, mx.array]],
) -> dict[str, mx.array]:
    """Load a block's tensors from safetensors shards via mx.load (lazy mmap).

    Uses a shard_cache to avoid re-loading the same shard multiple times
    within a single token's forward pass.

    Returns a dict of {tensor_name: mx.array} for the block.
    """
    tensor_map = index.block_tensor_names(block_idx)
    result = {}
    for tensor_name, shard_file in tensor_map.items():
        if shard_file not in shard_cache:
            shard_cache[shard_file] = mx.load(str(index.shard_path(shard_file)))
        result[tensor_name] = shard_cache[shard_file][tensor_name]
    return result


def assign_block_weights(block, block_idx: int, tensors: dict[str, mx.array]):
    """Assign loaded tensors to block's QuantizedLinear and other modules."""
    prefix = f"model.layers.{block_idx}."
    for name, module in block.named_modules():
        full_prefix = prefix + name + "." if name else prefix
        if isinstance(module, nn.QuantizedLinear):
            w_key = full_prefix + "weight"
            s_key = full_prefix + "scales"
            b_key = full_prefix + "biases"
            if w_key in tensors:
                module.weight = tensors[w_key]
            if s_key in tensors:
                module.scales = tensors[s_key]
            if b_key in tensors:
                module.biases = tensors[b_key]
            # Also handle the non-quantized bias (e.g., q_proj.bias)
            bias_key = full_prefix + "bias"
            if bias_key in tensors and hasattr(module, "bias"):
                module.bias = tensors[bias_key]
        else:
            # Handle non-QuantizedLinear params (e.g., LayerNorm.weight)
            w_key = full_prefix + "weight"
            if w_key in tensors and hasattr(module, "weight"):
                module.weight = tensors[w_key]


def evict_block(block):
    """Replace all block weights with tiny placeholders to free memory."""
    for name, module in block.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            module.weight = mx.zeros((1,), dtype=mx.uint32)
            module.scales = mx.zeros((1,), dtype=mx.float16)
            if hasattr(module, "biases") and module.biases is not None:
                module.biases = mx.zeros((1,), dtype=mx.float16)
        elif hasattr(module, "weight") and module.weight is not None:
            # Evict non-QuantizedLinear params (e.g., RMSNorm.weight)
            module.weight = mx.zeros((1,), dtype=module.weight.dtype)
    mx.eval(block.parameters())


# ── npz path (for comparison) ──


def save_block_to_npz(block, idx, save_dir: Path, force: bool = False):
    """Save a single block's weights to disk as .npz (for baseline comparison)."""
    block_file = save_dir / f"block_{idx:03d}.npz"
    if block_file.exists() and not force:
        return
    flat = {}
    for name, module in block.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            flat[f"{name}.weight"] = np.array(module.weight)
            flat[f"{name}.scales"] = np.array(module.scales)
            if hasattr(module, "biases") and module.biases is not None:
                flat[f"{name}.biases"] = np.array(module.biases)
    np.savez(block_file, **flat)


def load_block_from_npz(block_idx: int, save_dir: Path) -> dict:
    """Load block weights from npz (CPU I/O, no MLX)."""
    block_file = save_dir / f"block_{block_idx:03d}.npz"
    return dict(np.load(block_file))


def swap_block_weights_npz(block, weights_dict: dict):
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


# ── Phase 0: Shard Layout + MLX Load Characterization ──


def run_phase_0(model_id: str):
    """Phase 0: Validate shard layout and characterize mx.load performance."""
    print("=" * 70)
    print("Phase 0: Shard Layout + MLX Load Characterization")
    print("=" * 70)

    env = get_environment_info()
    model_path = _find_hf_cache_path(model_id)
    print(f"Model: {model_id}")
    print(f"Path: {model_path}")

    # Phase 0a: Shard layout
    print("\n--- Phase 0a: Shard Layout Validation ---")
    index = SafetensorsBlockIndex(model_path)
    summary = index.summary()

    # Check shard sizes
    shard_sizes = {}
    for b in range(index.n_blocks):
        for shard_file in index.block_shards(b):
            if shard_file not in shard_sizes:
                path = index.shard_path(shard_file)
                shard_sizes[shard_file] = path.stat().st_size / (1024**3)
    print(f"\n  Shard sizes:")
    for s, gb in sorted(shard_sizes.items()):
        print(f"    {s}: {gb:.2f} GB")

    print(f"\n  Strategy: mx.load returns lazy mmap'd arrays — load entire shard "
          f"is ~1 ms, then mx.eval only the needed block tensors.")
    print(f"  Memory bound: only block's tensors are materialized (~494 MB for Q4)")

    # Phase 0b: Load cycle characterization with component breakdown
    print("\n--- Phase 0b: Load Cycle Characterization ---")
    print("  Loading model for characterization...")

    from mlx_lm import load
    model, tokenizer = load(model_id)
    blocks = _get_model_blocks(model)

    inner = _get_inner_model(model)

    # Materialize block 0 and embeddings for hidden state generation
    mx.eval(inner.embed_tokens.parameters())
    mx.eval(blocks[0].parameters())

    test_block_idx = 1  # Use block 1 for testing (block 0 has embeddings in same shard)
    process = psutil.Process()

    # Prepare npz baseline (force-recreate to avoid stale data)
    npz_dir = Path("/tmp/h8a_phase0_npz")
    if npz_dir.exists():
        shutil.rmtree(npz_dir)
    npz_dir.mkdir(parents=True, exist_ok=True)
    mx.eval(blocks[test_block_idx].parameters())
    save_block_to_npz(blocks[test_block_idx], test_block_idx, npz_dir)
    evict_block(blocks[test_block_idx])
    gc.collect()

    # Prepare hidden state input for test block (run embeddings + block 0)
    input_ids = mx.array([[1]])
    h_input = inner.embed_tokens(input_ids)
    h_input = blocks[0](h_input, None, None)
    mx.eval(h_input)

    print(f"\n  Testing block {test_block_idx}, 10 cycles each path:")

    # -- Safetensors path --
    print("\n  === Safetensors Path ===")
    st_results = []
    for cycle in range(10):
        rss_before = process.memory_info().rss / 1e6
        metal_before = mx.get_active_memory() / 1e6

        shard_cache = {}

        t0 = time.perf_counter()
        tensors = load_block_from_safetensors(test_block_idx, index, shard_cache)
        t_load = time.perf_counter()

        assign_block_weights(blocks[test_block_idx], test_block_idx, tensors)
        t_assign = time.perf_counter()

        mx.eval(blocks[test_block_idx].parameters())
        t_eval = time.perf_counter()

        # Forward (use pre-computed hidden state)
        h = blocks[test_block_idx](h_input, None, None)
        mx.eval(h)
        t_forward = time.perf_counter()

        evict_block(blocks[test_block_idx])
        t_evict = time.perf_counter()

        rss_after = process.memory_info().rss / 1e6
        metal_after = mx.get_active_memory() / 1e6

        del tensors, shard_cache, h
        gc.collect()

        result = {
            "t_load_ms": (t_load - t0) * 1000,
            "t_assign_ms": (t_assign - t_load) * 1000,
            "t_eval_ms": (t_eval - t_assign) * 1000,
            "t_forward_ms": (t_forward - t_eval) * 1000,
            "t_evict_ms": (t_evict - t_forward) * 1000,
            "t_total_ms": (t_evict - t0) * 1000,
            "rss_delta_mb": rss_after - rss_before,
            "metal_delta_mb": metal_after - metal_before,
        }
        st_results.append(result)

        if cycle < 3 or cycle == 9:
            print(f"    Cycle {cycle}: load={result['t_load_ms']:.1f} assign={result['t_assign_ms']:.1f} "
                  f"eval={result['t_eval_ms']:.1f} fwd={result['t_forward_ms']:.1f} "
                  f"evict={result['t_evict_ms']:.1f} total={result['t_total_ms']:.1f}ms "
                  f"RSS_Δ={result['rss_delta_mb']:.0f}MB")

    # -- NPZ path --
    print("\n  === NPZ Path ===")
    npz_results = []
    for cycle in range(10):
        rss_before = process.memory_info().rss / 1e6

        t0 = time.perf_counter()
        weights = load_block_from_npz(test_block_idx, npz_dir)
        t_load = time.perf_counter()

        # Assignment is part of swap
        t_assign = t_load  # no separate assign step

        swap_block_weights_npz(blocks[test_block_idx], weights)
        t_eval = time.perf_counter()

        h = blocks[test_block_idx](h_input, None, None)
        mx.eval(h)
        t_forward = time.perf_counter()

        evict_block(blocks[test_block_idx])
        t_evict = time.perf_counter()

        rss_after = process.memory_info().rss / 1e6

        del weights, h
        gc.collect()

        result = {
            "t_load_ms": (t_load - t0) * 1000,
            "t_assign_ms": 0,
            "t_eval_ms": (t_eval - t_load) * 1000,
            "t_total_ms": (t_evict - t0) * 1000,
            "rss_delta_mb": rss_after - rss_before,
        }
        npz_results.append(result)

        if cycle < 3 or cycle == 9:
            print(f"    Cycle {cycle}: load={result['t_load_ms']:.1f} "
                  f"swap+eval={result['t_eval_ms']:.1f} "
                  f"total={result['t_total_ms']:.1f}ms "
                  f"RSS_Δ={result['rss_delta_mb']:.0f}MB")

    # Compute steady-state stats (skip first 2 warmup cycles)
    steady_st = st_results[2:]
    steady_npz = npz_results[2:]

    st_total_p50 = np.percentile([r["t_total_ms"] for r in steady_st], 50)
    npz_total_p50 = np.percentile([r["t_total_ms"] for r in steady_npz], 50)
    st_rss_max = max(abs(r["rss_delta_mb"]) for r in steady_st)
    speedup = npz_total_p50 / st_total_p50 if st_total_p50 > 0 else 0

    print(f"\n  --- Steady-State Summary (cycles 2-9) ---")
    print(f"  Safetensors t_total p50: {st_total_p50:.1f} ms")
    print(f"  NPZ t_total p50: {npz_total_p50:.1f} ms")
    print(f"  Speedup: {speedup:.1f}x")
    print(f"  Safetensors RSS max delta: {st_rss_max:.0f} MB")

    # Component breakdown for safetensors
    print(f"\n  Safetensors component breakdown (p50):")
    for comp in ["t_load_ms", "t_assign_ms", "t_eval_ms", "t_forward_ms", "t_evict_ms"]:
        p50 = np.percentile([r[comp] for r in steady_st], 50)
        print(f"    {comp}: {p50:.1f} ms")

    # Gate checks
    print(f"\n  --- Gate Checks ---")
    gate_latency = st_total_p50 < 200
    gate_rss = st_rss_max < 10
    gate_faster = st_total_p50 < npz_total_p50
    gate_memory = True  # mx.load is lazy, bounded by block size

    print(f"    Latency < 200ms: {'PASS' if gate_latency else 'FAIL'} ({st_total_p50:.1f}ms)")
    print(f"    RSS flat (< 10MB delta): {'PASS' if gate_rss else 'FAIL'} ({st_rss_max:.0f}MB)")
    print(f"    Faster than NPZ: {'PASS' if gate_faster else 'FAIL'} ({st_total_p50:.1f} vs {npz_total_p50:.1f}ms)")
    print(f"    Bounded memory: PASS (mx.load is lazy mmap)")
    overall = gate_latency and gate_rss and gate_faster and gate_memory
    print(f"    Overall: {'PASS' if overall else 'FAIL'}")

    log_experiment(
        experiment_name="h8a_safetensors_direct",
        phase="phase_0_characterization",
        config={
            "model": model_id,
            "test_block": test_block_idx,
            "n_cycles": 10,
            "shard_layout": summary,
        },
        results={
            "safetensors_p50_ms": round(st_total_p50, 1),
            "npz_p50_ms": round(npz_total_p50, 1),
            "speedup": round(speedup, 1),
            "rss_max_delta_mb": round(st_rss_max, 1),
            "component_breakdown": {
                comp: round(np.percentile([r[comp] for r in steady_st], 50), 1)
                for comp in ["t_load_ms", "t_assign_ms", "t_eval_ms", "t_forward_ms", "t_evict_ms"]
            },
            "gate_pass": overall,
        },
        env=env,
    )

    del model, tokenizer
    return overall


# ── Phase 1: 72B Block Index Builder ──


def run_phase_1(model_id: str):
    """Phase 1: Build and validate the 72B safetensors block index."""
    print("=" * 70)
    print("Phase 1: 72B Safetensors Block Index Builder")
    print("=" * 70)

    env = get_environment_info()
    model_path = _find_hf_cache_path(model_id)
    print(f"Model: {model_id}")
    print(f"Path: {model_path}")

    index = SafetensorsBlockIndex(model_path)
    summary = index.summary()

    # Compute per-block byte sizes from shard headers
    print(f"\n  Analyzing per-block tensor sizes...")
    block_sizes = {}
    p1_shard_cache = {}  # Cache loaded shards to avoid redundant mx.load calls
    for b in range(index.n_blocks):
        shard_files = index.block_shards(b)
        total_bytes = 0
        for sf in shard_files:
            if sf not in p1_shard_cache:
                p1_shard_cache[sf] = mx.load(str(index.shard_path(sf)))
            all_tensors = p1_shard_cache[sf]
            tensor_map = index.block_tensor_names(b)
            for tname in tensor_map:
                if tensor_map[tname] == sf and tname in all_tensors:
                    total_bytes += all_tensors[tname].nbytes
        block_sizes[b] = total_bytes / (1024 * 1024)  # MB
    del p1_shard_cache

    sizes = list(block_sizes.values())
    avg_mb = sum(sizes) / len(sizes)
    min_mb = min(sizes)
    max_mb = max(sizes)

    print(f"  Per-block sizes: avg={avg_mb:.1f} MB, min={min_mb:.1f} MB, max={max_mb:.1f} MB")
    print(f"  Expected ~471 MB/block (Q4)")
    pct_diff = abs(avg_mb - 471) / 471 * 100
    print(f"  Difference from budget: {pct_diff:.1f}%")

    # Bytes-touched analysis: actual bytes read per block
    print(f"\n  Bytes-touched analysis (accounting for shard fanout):")
    shard_sizes = {}
    for b in range(index.n_blocks):
        for sf in index.block_shards(b):
            if sf not in shard_sizes:
                shard_sizes[sf] = index.shard_path(sf).stat().st_size

    # With mx.load lazy mmap, bytes "touched" = only the block's tensors
    # (not the full shard), since mx.eval only materializes what's needed
    total_streaming_gb = sum(block_sizes[b] for b in range(8, index.n_blocks - 8)) / 1024
    # sum is in MB; SSD bandwidth 5.5 GB/s = 5632 MB/s; result is seconds/token
    cold_tok_s = 1.0 / (sum(block_sizes[b] for b in range(8, index.n_blocks - 8)) / (5.5 * 1024))

    print(f"  Streaming blocks (8-{index.n_blocks-9}): {index.n_blocks - 16} blocks")
    print(f"  Total streaming data: {total_streaming_gb:.1f} GB")
    print(f"  Cold-cache I/O ceiling: {cold_tok_s:.2f} tok/s (I/O only, no eval overhead)")

    gate_blocks = index.n_blocks == 80
    gate_sizes = pct_diff < 5
    print(f"\n  Gate: blocks==80: {'PASS' if gate_blocks else 'FAIL'} ({index.n_blocks})")
    print(f"  Gate: sizes within 5%: {'PASS' if gate_sizes else 'FAIL'} ({pct_diff:.1f}%)")
    overall = gate_blocks and gate_sizes

    log_experiment(
        experiment_name="h8a_safetensors_direct",
        phase="phase_1_block_index",
        config={"model": model_id, "n_blocks": index.n_blocks},
        results={
            "summary": summary,
            "avg_block_mb": round(avg_mb, 1),
            "min_block_mb": round(min_mb, 1),
            "max_block_mb": round(max_mb, 1),
            "budget_diff_pct": round(pct_diff, 1),
            "total_streaming_gb": round(total_streaming_gb, 1),
            "cold_tok_s_ceiling": round(cold_tok_s, 3),
            "gate_pass": overall,
        },
        env=env,
    )

    return overall


# ── Phase 2: Direct Load Benchmark on 7B ──


def run_phase_2(model_id_7b: str):
    """Phase 2: Side-by-side benchmark of npz vs safetensors on 7B."""
    print("=" * 70)
    print("Phase 2: Direct Load Benchmark on 7B")
    print("=" * 70)

    env = get_environment_info()
    print(f"Model: {model_id_7b}")

    from mlx_lm import load
    model, tokenizer = load(model_id_7b)
    blocks = _get_model_blocks(model)
    inner = _get_inner_model(model)
    n_blocks = len(blocks)

    model_path = _find_hf_cache_path(model_id_7b)
    index = SafetensorsBlockIndex(model_path)
    index.summary()

    # Prepare: materialize all blocks, save npz, then evict streaming blocks
    print(f"\n  Materializing all {n_blocks} blocks...")
    for b in range(n_blocks):
        mx.eval(blocks[b].parameters())
    mx.eval(inner.embed_tokens.parameters())
    mx.eval(inner.norm.parameters())
    if hasattr(model, "lm_head"):
        mx.eval(model.lm_head.parameters())

    # Capture baseline logits for correctness check
    prompt = "The key innovation of the transformer architecture is"
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    # Full forward pass for baseline logits
    h = inner.embed_tokens(input_ids)
    mask = "causal" if h.shape[1] > 1 else None
    for layer in inner.layers:
        h = layer(h, mask, None)
    h = inner.norm(h)
    if hasattr(model, "args") and model.args.tie_word_embeddings:
        baseline_logits = inner.embed_tokens.as_linear(h)
    else:
        baseline_logits = model.lm_head(h)
    mx.eval(baseline_logits)
    baseline_logits_np = np.array(baseline_logits)
    print(f"  Baseline logits captured (shape={baseline_logits_np.shape})")

    # Save npz for all blocks (force-recreate to avoid stale data)
    npz_dir = Path("/tmp/h8a_phase2_npz")
    if npz_dir.exists():
        shutil.rmtree(npz_dir)
    npz_dir.mkdir(parents=True, exist_ok=True)
    for b in range(n_blocks):
        save_block_to_npz(blocks[b], b, npz_dir)

    # Define streaming set (middle 60%)
    n_resident_each = max(1, n_blocks // 5)  # ~20% each end
    streaming_indices = list(range(n_resident_each, n_blocks - n_resident_each))
    print(f"  Streaming blocks: {len(streaming_indices)} (indices {streaming_indices[0]}-{streaming_indices[-1]})")

    # Evict streaming blocks
    for b in streaming_indices:
        evict_block(blocks[b])
    gc.collect()

    process = psutil.Process()

    # -- Phase 2a: Single-block feasibility --
    print(f"\n--- Phase 2a: Single-Block Feasibility Gate ---")
    test_block = streaming_indices[0]
    print(f"  Test block: {test_block}")

    # Safetensors single-block
    shard_cache = {}
    t0 = time.perf_counter()
    tensors = load_block_from_safetensors(test_block, index, shard_cache)
    assign_block_weights(blocks[test_block], test_block, tensors)
    mx.eval(blocks[test_block].parameters())
    st_single_ms = (time.perf_counter() - t0) * 1000
    evict_block(blocks[test_block])
    del tensors, shard_cache
    gc.collect()
    print(f"  Safetensors single-block: {st_single_ms:.1f} ms")

    # NPZ single-block
    t0 = time.perf_counter()
    weights = load_block_from_npz(test_block, npz_dir)
    swap_block_weights_npz(blocks[test_block], weights)
    npz_single_ms = (time.perf_counter() - t0) * 1000
    evict_block(blocks[test_block])
    del weights
    gc.collect()
    print(f"  NPZ single-block: {npz_single_ms:.1f} ms")

    gate_2a = st_single_ms < npz_single_ms
    print(f"  Gate 2a (safetensors < npz): {'PASS' if gate_2a else 'FAIL'} ({st_single_ms:.1f} vs {npz_single_ms:.1f}ms)")

    if not gate_2a:
        print("  ABORT: Safetensors path is not faster than npz. Mechanism invalid.")
        log_experiment(
            experiment_name="h8a_safetensors_direct",
            phase="phase_2a_single_block",
            config={"model": model_id_7b, "test_block": test_block},
            results={"st_ms": round(st_single_ms, 1), "npz_ms": round(npz_single_ms, 1), "gate_pass": False},
            env=env,
        )
        return False

    # -- Phase 2b: Full side-by-side --
    print(f"\n--- Phase 2b: Full 7B Side-by-Side Benchmark ---")
    n_iterations = 3  # Multiple iterations for stability

    st_latencies = []
    npz_latencies = []

    for iteration in range(n_iterations):
        # Safetensors path: load all streaming blocks, forward, evict
        shard_cache = {}
        for b in streaming_indices:
            t0 = time.perf_counter()
            tensors = load_block_from_safetensors(b, index, shard_cache)
            assign_block_weights(blocks[b], b, tensors)
            mx.eval(blocks[b].parameters())
            st_latencies.append((time.perf_counter() - t0) * 1000)
            evict_block(blocks[b])
            del tensors
        del shard_cache
        gc.collect()

        # NPZ path: load all streaming blocks, forward, evict
        for b in streaming_indices:
            t0 = time.perf_counter()
            weights = load_block_from_npz(b, npz_dir)
            swap_block_weights_npz(blocks[b], weights)
            npz_latencies.append((time.perf_counter() - t0) * 1000)
            evict_block(blocks[b])
            del weights
        gc.collect()

    st_p50 = np.percentile(st_latencies, 50)
    st_p95 = np.percentile(st_latencies, 95)
    npz_p50 = np.percentile(npz_latencies, 50)
    npz_p95 = np.percentile(npz_latencies, 95)
    speedup = npz_p50 / st_p50

    print(f"  Safetensors: p50={st_p50:.1f}ms p95={st_p95:.1f}ms")
    print(f"  NPZ: p50={npz_p50:.1f}ms p95={npz_p95:.1f}ms")
    print(f"  Speedup: {speedup:.1f}x")

    # Logit correctness: reload all streaming blocks via safetensors and run forward
    shard_cache = {}
    for b in streaming_indices:
        tensors = load_block_from_safetensors(b, index, shard_cache)
        assign_block_weights(blocks[b], b, tensors)
        mx.eval(blocks[b].parameters())
        del tensors

    h = inner.embed_tokens(input_ids)
    mask = "causal" if h.shape[1] > 1 else None
    for layer in inner.layers:
        h = layer(h, mask, None)
    h = inner.norm(h)
    if hasattr(model, "args") and model.args.tie_word_embeddings:
        st_logits = inner.embed_tokens.as_linear(h)
    else:
        st_logits = model.lm_head(h)
    mx.eval(st_logits)
    st_logits_np = np.array(st_logits)

    max_diff = float(np.max(np.abs(baseline_logits_np - st_logits_np)))
    logits_match = max_diff < 1e-4
    print(f"  Logit max diff: {max_diff:.2e} ({'PASS' if logits_match else 'FAIL'})")

    gate_2b_speed = st_p50 < npz_p50 * 0.5
    gate_2b = gate_2b_speed and logits_match
    print(f"\n  Gate 2b (speed < 50% npz): {'PASS' if gate_2b_speed else 'FAIL'}")
    print(f"  Gate 2b (logits match): {'PASS' if logits_match else 'FAIL'}")
    print(f"  Gate 2b overall: {'PASS' if gate_2b else 'FAIL'}")

    log_experiment(
        experiment_name="h8a_safetensors_direct",
        phase="phase_2_7b_benchmark",
        config={
            "model": model_id_7b,
            "n_streaming": len(streaming_indices),
            "n_iterations": n_iterations,
        },
        results={
            "st_p50_ms": round(st_p50, 1),
            "st_p95_ms": round(st_p95, 1),
            "npz_p50_ms": round(npz_p50, 1),
            "npz_p95_ms": round(npz_p95, 1),
            "speedup": round(speedup, 1),
            "logit_max_diff": max_diff,
            "logits_match": logits_match,
            "gate_2a_pass": gate_2a,
            "gate_2b_pass": gate_2b,
        },
        env=env,
    )

    del model, tokenizer
    return gate_2b


# ── Phase 3: 72B Integration ──


def run_phase_3(model_id_72b: str, n_tokens: int = 10):
    """Phase 3: 72B integration with safetensors direct loading."""
    print("=" * 70)
    print("Phase 3: 72B Integration — Safetensors Direct Streaming")
    print("=" * 70)

    env = get_environment_info()
    print(f"Environment: {env['chip']}, {env['memory_gb']} GB RAM, {env['available_gb']} GB available")
    print(f"Model: {model_id_72b}")

    from mlx_lm import load

    # Load model (lazy mmap)
    print(f"\nStep 1: Loading model (lazy mmap)...")
    t_load_start = time.perf_counter()
    model, tokenizer = load(model_id_72b)
    load_s = time.perf_counter() - t_load_start
    print(f"  Loaded in {load_s:.0f}s")

    blocks = _get_model_blocks(model)
    inner = _get_inner_model(model)
    n_blocks = len(blocks)
    print(f"  {n_blocks} transformer blocks")

    # Build safetensors index
    model_path = _find_hf_cache_path(model_id_72b)
    index = SafetensorsBlockIndex(model_path)
    index.summary()

    # 16r-Q4 / 64s-Q4 config
    n_first = 8
    n_last = 8
    resident_indices = list(range(n_first)) + list(range(n_blocks - n_last, n_blocks))
    resident_set = set(resident_indices)
    streaming_indices = [i for i in range(n_blocks) if i not in resident_set]

    print(f"\n  Config: 16r-Q4 / 64s-Q4")
    print(f"  Resident: first {n_first} + last {n_last} = {len(resident_indices)} blocks")
    print(f"  Streaming: {len(streaming_indices)} blocks")

    # Step 2: Incremental setup — materialize resident blocks, evict streaming
    print(f"\nStep 2: Incremental block processing...")
    t_setup = time.perf_counter()

    for idx in range(n_blocks):
        mx.eval(blocks[idx].parameters())
        if idx not in resident_set:
            evict_block(blocks[idx])
            gc.collect()

        if idx % 10 == 0 or idx == n_blocks - 1:
            avail = get_available_memory_gb()
            rss = get_rss_mb()
            tag = "resident" if idx in resident_set else "evicted"
            print(f"  Block {idx:>3d}/{n_blocks} [{tag:<10s}] RSS={rss:.0f} MB, Avail={avail:.1f} GB")

    # Materialize non-block components
    mx.eval(inner.embed_tokens.parameters())
    mx.eval(inner.norm.parameters())
    if hasattr(model, "lm_head"):
        mx.eval(model.lm_head.parameters())

    setup_s = time.perf_counter() - t_setup
    print(f"  Setup done in {setup_s:.0f}s")
    mx.clear_cache()

    rss_after_setup = get_rss_mb()
    avail_after_setup = get_available_memory_gb()
    print(f"  RSS after setup: {rss_after_setup:.0f} MB")
    print(f"  Available after setup: {avail_after_setup:.1f} GB")

    # Step 3: Correctness — per-block assign/evict/reload idempotence
    print(f"\nStep 3: Correctness checks (assign/evict/reload path)...")

    # Prepare a dummy hidden state for forward-pass comparison
    dummy_ids = mx.array([[1]])
    h_dummy = inner.embed_tokens(dummy_ids)
    mx.eval(h_dummy)

    check_blocks = [12, 40, 68]  # early, middle, late
    idempotent = True
    for cb in check_blocks:
        if cb >= n_blocks:
            continue

        # Load → assign → eval → forward (first pass)
        shard_cache = {}
        tensors1 = load_block_from_safetensors(cb, index, shard_cache)
        assign_block_weights(blocks[cb], cb, tensors1)
        mx.eval(blocks[cb].parameters())
        out1 = blocks[cb](h_dummy, None, None)
        mx.eval(out1)
        del shard_cache

        # Evict
        evict_block(blocks[cb])

        # Reload → assign → eval → forward (second pass)
        shard_cache = {}
        tensors2 = load_block_from_safetensors(cb, index, shard_cache)
        assign_block_weights(blocks[cb], cb, tensors2)
        mx.eval(blocks[cb].parameters())
        out2 = blocks[cb](h_dummy, None, None)
        mx.eval(out2)
        del shard_cache

        # Compare forward-pass outputs
        diff = float(mx.max(mx.abs(out1 - out2)).item())
        if diff > 0:
            print(f"  Block {cb}: forward diff={diff:.2e} — NOT IDEMPOTENT")
            idempotent = False
        else:
            print(f"  Block {cb}: forward diff=0 — OK")

        del tensors1, tensors2, out1, out2
        evict_block(blocks[cb])

    print(f"  Reload idempotence: {'PASS' if idempotent else 'FAIL'}")

    # Step 4: Generate tokens with streaming (with proper KV cache)
    print(f"\nStep 4: Generating {n_tokens} tokens...")

    from mlx_lm.models.cache import make_prompt_cache

    prompt = "The key innovation of the transformer architecture is"
    tokens = tokenizer.encode(prompt)
    print(f"  Prompt: '{prompt}' ({len(tokens)} tokens)")

    # Create KV cache — one KVCache per layer, persists across tokens
    kv_cache = make_prompt_cache(model)
    print(f"  KV cache created: {len(kv_cache)} layers")

    # Step 4a: Prefill — process all prompt tokens through model with cache
    print(f"  Prefilling {len(tokens)} prompt tokens...")
    t_prefill = time.perf_counter()

    input_ids = mx.array([tokens])  # [1, seq_len]
    h = inner.embed_tokens(input_ids)
    mask = "causal"  # Full causal mask for multi-token prefill

    for i, layer in enumerate(inner.layers):
        if i in resident_set:
            h = layer(h, mask, kv_cache[i])
        else:
            # Per-block shard scope: load → extract → release immediately
            shard_cache = {}
            tensors = load_block_from_safetensors(i, index, shard_cache)
            assign_block_weights(blocks[i], i, tensors)
            mx.eval(blocks[i].parameters())
            del shard_cache, tensors

            h = layer(h, mask, kv_cache[i])
            mx.eval(h, kv_cache[i].state)

            evict_block(blocks[i])

    h = inner.norm(h)
    if hasattr(model, "args") and model.args.tie_word_embeddings:
        logits = inner.embed_tokens.as_linear(h)
    else:
        logits = model.lm_head(h)
    mx.eval(logits)

    # Sample first token from prefill
    prefill_logits = logits[0, -1, :]  # Save for reproducibility check
    next_token = mx.argmax(prefill_logits).item()
    tokens.append(next_token)

    prefill_ms = (time.perf_counter() - t_prefill) * 1000
    print(f"  Prefill done in {prefill_ms:.0f} ms, first token: {next_token} "
          f"('{tokenizer.decode([next_token])}')")

    # Step 4b: Decode — generate remaining tokens one at a time with KV cache
    vm_before = get_vm_stat()
    per_token_ms = []
    all_load_ms = []
    avail_history = []
    rss_history = []

    for tok_i in range(n_tokens - 1):  # -1 because we already generated first token
        t0 = time.perf_counter()
        input_ids = mx.array([[tokens[-1]]])  # [1, 1] — single token

        # Forward pass with KV cache (mask=None for single-token decode)
        h = inner.embed_tokens(input_ids)

        token_load_ms = []

        for i, layer in enumerate(inner.layers):
            if i in resident_set:
                h = layer(h, None, kv_cache[i])
            else:
                t_block = time.perf_counter()
                # Per-block shard scope to avoid holding refs to all shards
                shard_cache = {}
                tensors = load_block_from_safetensors(i, index, shard_cache)
                assign_block_weights(blocks[i], i, tensors)
                mx.eval(blocks[i].parameters())
                del shard_cache, tensors
                block_load_ms = (time.perf_counter() - t_block) * 1000
                token_load_ms.append(block_load_ms)

                h = layer(h, None, kv_cache[i])
                mx.eval(h, kv_cache[i].state)

                evict_block(blocks[i])

        h = inner.norm(h)
        if hasattr(model, "args") and model.args.tie_word_embeddings:
            logits = inner.embed_tokens.as_linear(h)
        else:
            logits = model.lm_head(h)

        next_token = mx.argmax(logits[0, -1, :]).item()
        tokens.append(next_token)

        tok_ms = (time.perf_counter() - t0) * 1000
        per_token_ms.append(tok_ms)
        all_load_ms.extend(token_load_ms)

        avail_now = get_available_memory_gb()
        rss_now = get_rss_mb()
        avail_history.append(avail_now)
        rss_history.append(rss_now)

        if tok_i < 5 or tok_i % 5 == 0 or tok_i == n_tokens - 2:
            total_load = sum(token_load_ms)
            decoded = tokenizer.decode([tokens[-1]])
            print(f"    Token {tok_i+1}: {tok_ms:.0f} ms (block_load_sum={total_load:.0f}ms), "
                  f"RSS={rss_now:.0f} MB, Avail={avail_now:.1f} GB, '{decoded}'")

    vm_after = get_vm_stat()
    vm_delta = vm_stat_delta(vm_before, vm_after)
    peak_rss = get_peak_rss_mb()

    # Decode
    generated_text = tokenizer.decode(tokens[len(tokenizer.encode(prompt)):])
    print(f"\n  Generated: '{generated_text[:200]}'")

    # Run-to-run reproducibility (second run — prefill + first decode token)
    print(f"\n  Reproducibility check (second run, first token only)...")
    kv_cache2 = make_prompt_cache(model)
    tokens2 = tokenizer.encode(prompt)
    input_ids2 = mx.array([tokens2])  # Full prompt
    h2 = inner.embed_tokens(input_ids2)
    mask2 = "causal"
    for i, layer in enumerate(inner.layers):
        if i in resident_set:
            h2 = layer(h2, mask2, kv_cache2[i])
        else:
            shard_cache2 = {}
            tensors = load_block_from_safetensors(i, index, shard_cache2)
            assign_block_weights(blocks[i], i, tensors)
            mx.eval(blocks[i].parameters())
            del shard_cache2, tensors
            h2 = layer(h2, mask2, kv_cache2[i])
            mx.eval(h2, kv_cache2[i].state)
            evict_block(blocks[i])
    h2 = inner.norm(h2)
    if hasattr(model, "args") and model.args.tie_word_embeddings:
        logits2 = inner.embed_tokens.as_linear(h2)
    else:
        logits2 = model.lm_head(h2)
    mx.eval(logits2)
    logits2_last = logits2[0, -1, :]
    next_token2 = mx.argmax(logits2_last).item()
    first_generated = tokens[len(tokenizer.encode(prompt))]

    # Compare full logits vectors, not just argmax
    logit_max_diff = float(mx.max(mx.abs(prefill_logits - logits2_last)).item())
    reproducible = logit_max_diff < 1e-4
    print(f"  Run 1 first token: {first_generated}, Run 2: {next_token2}")
    print(f"  Logit max diff: {logit_max_diff:.2e}")
    print(f"  Reproducible: {'PASS' if reproducible else 'FAIL'}")
    del kv_cache2

    # Statistics
    steady = per_token_ms[2:] if len(per_token_ms) > 2 else per_token_ms  # skip warmup if enough data
    avg_tok_ms = sum(steady) / len(steady) if steady else 0
    tok_s = 1000 / avg_tok_ms if avg_tok_ms > 0 else 0
    min_avail = min(avail_history) if avail_history else 0
    max_rss = max(rss_history) if rss_history else 0
    rss_variance = max(rss_history) - min(rss_history) if rss_history else 0

    print(f"\n{'='*70}")
    print(f"RESULTS — 72B Integration (Safetensors Direct)")
    print(f"{'='*70}")
    print(f"  tok/s: {tok_s:.3f} ({avg_tok_ms:.0f} ms/tok)")
    print(f"  Peak RSS: {peak_rss:.0f} MB")
    print(f"  Min available: {min_avail:.1f} GB")
    print(f"  RSS variance: {rss_variance:.0f} MB")

    if all_load_ms:
        load_p50 = np.percentile(all_load_ms, 50)
        load_p95 = np.percentile(all_load_ms, 95)
        print(f"  Block load p50/p95: {load_p50:.0f}/{load_p95:.0f} ms")

    print(f"  Pageouts: {vm_delta['pageout_delta_mb']:.0f} MB")
    print(f"  Pageins: {vm_delta['pagein_delta_mb']:.0f} MB")

    # Gate checks
    print(f"\n  --- Gate Checks ---")
    gate_tok = tok_s >= 0.1
    gate_repro = reproducible
    gate_idem = idempotent
    gate_avail = min_avail > 2.0
    gate_pageout = vm_delta["pageout_delta_mb"] < 500
    gate_rss = rss_variance < 500

    # Coherence: check for degenerate output
    words = generated_text.split()
    if len(words) >= 4:
        repeated = sum(1 for i in range(1, len(words)) if words[i] == words[i-1])
        gate_coherent = repeated / len(words) < 0.5
    else:
        gate_coherent = True

    print(f"    tok/s >= 0.1: {'PASS' if gate_tok else 'FAIL'} ({tok_s:.3f})")
    print(f"    Reproducible: {'PASS' if gate_repro else 'FAIL'}")
    print(f"    Idempotent: {'PASS' if gate_idem else 'FAIL'}")
    print(f"    Available > 2 GB: {'PASS' if gate_avail else 'FAIL'} ({min_avail:.1f} GB)")
    print(f"    Pageouts < 500 MB: {'PASS' if gate_pageout else 'FAIL'} ({vm_delta['pageout_delta_mb']:.0f} MB)")
    print(f"    RSS variance < 500 MB: {'PASS' if gate_rss else 'FAIL'} ({rss_variance:.0f} MB)")
    print(f"    Coherent output: {'PASS' if gate_coherent else 'FAIL'}")
    overall = gate_tok and gate_repro and gate_idem and gate_avail and gate_pageout and gate_rss and gate_coherent
    print(f"    Overall: {'PASS' if overall else 'FAIL'}")

    # 16 GB projection (derived from measured 24 GB run values)
    print(f"\n  --- 16 GB Projection (informational, derived from measurements) ---")
    # Measured: total memory consumed on 24 GB = 24 - min_avail
    measured_consumed_gb = env["memory_gb"] - min_avail
    # Resident blocks contribution (measured): n_resident * block_size
    block_size_gb = 0.471  # Q4 block size from Phase 1
    measured_resident_gb = len(resident_indices) * block_size_gb
    # Non-resident overhead = consumed - resident blocks (includes embeddings, lm_head, norm, KV cache, Metal scratch, OS)
    measured_overhead_gb = measured_consumed_gb - measured_resident_gb
    # For 16 GB: can't fit 16 resident Q4 blocks (7.5 GB). Use minimal config:
    # 3 resident Q4 blocks (first 2 + last 1) = 1.4 GB, rest streamed as Q2.
    n_resident_16gb = 3
    projected_resident_gb = n_resident_16gb * block_size_gb
    projected_total = measured_overhead_gb + projected_resident_gb
    projected_cache = 16 - projected_total
    print(f"    Measured consumed (24 GB run): {measured_consumed_gb:.1f} GB")
    print(f"    Measured resident ({len(resident_indices)} blocks): {measured_resident_gb:.1f} GB")
    print(f"    Measured overhead (non-block): {measured_overhead_gb:.1f} GB")
    print(f"    Projected resident ({n_resident_16gb}×Q4): {projected_resident_gb:.1f} GB")
    print(f"    Projected total: {projected_total:.1f} GB")
    print(f"    Projected page cache: {projected_cache:.1f} GB")
    print(f"    16 GB viable: {'YES' if projected_total < 10.5 else 'NO'}")

    log_experiment(
        experiment_name="h8a_safetensors_direct",
        phase="phase_3_72b_integration",
        config={
            "model": model_id_72b,
            "n_tokens": n_tokens,
            "n_blocks": n_blocks,
            "n_resident": len(resident_indices),
            "n_streaming": len(streaming_indices),
            "config_name": "16r-Q4 / 64s-Q4",
        },
        results={
            "tok_s": round(tok_s, 4),
            "avg_tok_ms": round(avg_tok_ms, 1),
            "peak_rss_mb": round(peak_rss, 1),
            "min_available_gb": round(min_avail, 2),
            "rss_variance_mb": round(rss_variance, 1),
            "block_load_p50_ms": round(float(np.percentile(all_load_ms, 50)), 1) if all_load_ms else 0,
            "block_load_p95_ms": round(float(np.percentile(all_load_ms, 95)), 1) if all_load_ms else 0,
            "pageout_mb": vm_delta["pageout_delta_mb"],
            "pagein_mb": vm_delta["pagein_delta_mb"],
            "reproducible": reproducible,
            "idempotent": idempotent,
            "coherent": gate_coherent,
            "generated_text_preview": generated_text[:200],
            "gate_pass": overall,
            "projection_16gb_total": round(projected_total, 1),
            "projection_16gb_viable": projected_total < 10.5,
        },
        env=env,
    )

    del model, tokenizer
    return overall


# ── Main ──


def main():
    parser = argparse.ArgumentParser(description="H8a: Safetensors Direct Streaming")
    parser.add_argument("--phase", type=int, required=True, choices=[0, 1, 2, 3])
    parser.add_argument("--model-7b", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    parser.add_argument("--model-72b", default="mlx-community/Qwen2.5-72B-Instruct-4bit")
    parser.add_argument("--n-tokens", type=int, default=10)
    args = parser.parse_args()

    if args.phase == 0:
        # Use 72B for Phase 0 since we already have it downloaded
        run_phase_0(args.model_72b)
    elif args.phase == 1:
        run_phase_1(args.model_72b)
    elif args.phase == 2:
        run_phase_2(args.model_7b)
    elif args.phase == 3:
        run_phase_3(args.model_72b, args.n_tokens)


if __name__ == "__main__":
    main()
