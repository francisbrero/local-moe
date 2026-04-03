"""
H8b: Q2 Streaming with Cache Optimization — Performance Validation.

Builds on H8a's safetensors direct streaming infrastructure.
Quantizes streaming blocks from Q4 to Q2, shrinking the cyclic working set
from 29.4 GB to 16.8 GB, then applies cache optimization to improve tok/s.

Phases:
  0   - Q2 block micro-benchmark (pre-gate)
  0b  - Q2 quality pilot (early kill switch)
  1   - Q2 checkpoint preparation (quantize 64 streaming blocks)
  2   - Mixed-precision streaming integration (Q4 resident + Q2 streamed)
  3   - Cache optimization (madvise, readahead)
  4   - Quality validation (PPL comparison)
  5   - 16 GB provisional projection

Usage:
    uv run python scripts/q2_streaming_cache_opt.py --phase 0
    uv run python scripts/q2_streaming_cache_opt.py --phase 0b
    uv run python scripts/q2_streaming_cache_opt.py --phase 1
    uv run python scripts/q2_streaming_cache_opt.py --phase 2 [--n-tokens N]
    uv run python scripts/q2_streaming_cache_opt.py --phase 3 [--n-tokens N]
    uv run python scripts/q2_streaming_cache_opt.py --phase 4
    uv run python scripts/q2_streaming_cache_opt.py --phase 5 [--n-tokens N]
"""

import argparse
import ctypes
import gc
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
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
    create_memory_pressure,
)
from scripts.safetensors_direct_stream import (
    SafetensorsBlockIndex,
    load_block_from_safetensors,
    assign_block_weights,
    evict_block,
    _find_hf_cache_path,
    _get_model_blocks,
    _get_inner_model,
)

# ── Constants ──

Q2_CACHE_DIR = Path.home() / ".cache" / "local-moe" / "q2-blocks"
Q2_BITS = 2
Q2_GROUP_SIZE = 64
Q4_BITS = 4
Q4_GROUP_SIZE = 128
BLOCKS_PER_SHARD = 13  # ~3.4 GB per shard

# ── mincore utilities ──

libc = ctypes.CDLL("libSystem.B.dylib")
libc.mmap.restype = ctypes.c_void_p
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
PROT_READ = 1
MAP_PRIVATE = 2


def measure_file_residency(file_path: str, offset: int = 0, length: int = 0) -> float:
    """Measure page cache residency of a file region via mincore().

    Returns fraction [0, 1] of pages resident in page cache.
    If length=0, measures the entire file.
    """
    fd = os.open(file_path, os.O_RDONLY)
    try:
        file_size = os.fstat(fd).st_size
        if length == 0:
            length = file_size
        # mmap the file region
        addr = libc.mmap(
            None, ctypes.c_size_t(length), ctypes.c_int(PROT_READ),
            ctypes.c_int(MAP_PRIVATE), ctypes.c_int(fd), ctypes.c_long(offset),
        )
        if addr == ctypes.c_void_p(-1).value:
            return 0.0
        try:
            n_pages = (length + PAGE_SIZE - 1) // PAGE_SIZE
            vec = (ctypes.c_char * n_pages)()
            ret = libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(length), vec)
            if ret != 0:
                return 0.0
            resident = sum(1 for i in range(n_pages) if vec[i] != b'\x00')
            return resident / n_pages if n_pages > 0 else 0.0
        finally:
            libc.munmap(ctypes.c_void_p(addr), ctypes.c_size_t(length))
    finally:
        os.close(fd)


# ── Q2 quantization utilities ──


def dequantize_block(block, block_idx: int) -> dict[str, mx.array]:
    """Dequantize all QuantizedLinear layers in a block from Q4 to float16."""
    prefix = f"model.layers.{block_idx}."
    result = {}
    for name, module in block.named_modules():
        full_prefix = prefix + name + "." if name else prefix
        if isinstance(module, nn.QuantizedLinear):
            # Dequantize: packed uint32 -> float
            w_fp = mx.dequantize(
                module.weight, module.scales, module.biases,
                group_size=module.group_size, bits=module.bits,
            )
            # Re-quantize to Q2
            w_q2, s_q2, b_q2 = mx.quantize(w_fp, group_size=Q2_GROUP_SIZE, bits=Q2_BITS)
            mx.eval(w_q2, s_q2, b_q2)
            result[full_prefix + "weight"] = w_q2
            result[full_prefix + "scales"] = s_q2
            result[full_prefix + "biases"] = b_q2
            # Also handle non-quantized bias (e.g., q_proj.bias)
            if hasattr(module, "bias") and module.bias is not None:
                result[full_prefix + "bias"] = module.bias
            del w_fp
        else:
            # Non-QuantizedLinear params (RMSNorm.weight, etc.) — keep as-is
            if hasattr(module, "weight") and module.weight is not None:
                result[full_prefix + "weight"] = module.weight
    return result


def assign_q2_block_weights(block, block_idx: int, tensors: dict[str, mx.array]):
    """Assign Q2 tensors to a block, updating bits/group_size on QuantizedLinear modules."""
    prefix = f"model.layers.{block_idx}."
    for name, module in block.named_modules():
        full_prefix = prefix + name + "." if name else prefix
        if isinstance(module, nn.QuantizedLinear):
            w_key = full_prefix + "weight"
            s_key = full_prefix + "scales"
            b_key = full_prefix + "biases"
            if w_key in tensors:
                # Update bits/group_size before assigning weights
                module.bits = Q2_BITS
                module.group_size = Q2_GROUP_SIZE
                module.weight = tensors[w_key]
            if s_key in tensors:
                module.scales = tensors[s_key]
            if b_key in tensors:
                module.biases = tensors[b_key]
            bias_key = full_prefix + "bias"
            if bias_key in tensors and hasattr(module, "bias"):
                module.bias = tensors[bias_key]
        else:
            w_key = full_prefix + "weight"
            if w_key in tensors and hasattr(module, "weight"):
                module.weight = tensors[w_key]


def restore_q4_block_metadata(block):
    """Restore Q4 bits/group_size on QuantizedLinear modules after Q2 eviction."""
    for _, module in block.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            module.bits = Q4_BITS
            module.group_size = Q4_GROUP_SIZE


def save_q2_block_to_shard(
    block_tensors: dict[str, mx.array],
    shard_tensors: dict[str, np.ndarray],
):
    """Add a block's Q2 tensors to a shard dict (numpy arrays for safetensors)."""
    for name, arr in block_tensors.items():
        mx.eval(arr)
        shard_tensors[name] = np.array(arr)


def write_shard(shard_tensors: dict[str, np.ndarray], shard_path: Path):
    """Write a shard dict to a safetensors file."""
    from safetensors.numpy import save_file
    save_file(shard_tensors, str(shard_path))


def get_q2_block_size_mb(tensors: dict[str, mx.array]) -> float:
    """Compute total size of Q2 block tensors in MB."""
    total = sum(arr.nbytes for arr in tensors.values())
    return total / (1024 * 1024)


# ── Q2 Block Index ──


class Q2BlockIndex:
    """Maps streaming block indices to Q2 safetensors shards."""

    def __init__(self, q2_dir: Path):
        self.q2_dir = q2_dir
        self._block_map: dict[int, dict[str, str]] = defaultdict(dict)
        self._build_index()

    def _build_index(self):
        index_path = self.q2_dir / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"No Q2 index at {index_path}")
        with open(index_path) as f:
            data = json.load(f)
        for tensor_name, shard_file in data.get("weight_map", {}).items():
            m = re.match(r"model\.layers\.(\d+)\.", tensor_name)
            if m:
                block_idx = int(m.group(1))
                self._block_map[block_idx][tensor_name] = shard_file

    def has_block(self, block_idx: int) -> bool:
        return block_idx in self._block_map

    def block_tensor_names(self, block_idx: int) -> dict[str, str]:
        return dict(self._block_map[block_idx])

    def block_shards(self, block_idx: int) -> set[str]:
        return set(self._block_map[block_idx].values())

    def shard_path(self, shard_file: str) -> Path:
        return self.q2_dir / shard_file


def load_q2_block(
    block_idx: int,
    q2_index: Q2BlockIndex,
    shard_cache: dict[str, dict[str, mx.array]],
) -> dict[str, mx.array]:
    """Load a Q2 block's tensors from safetensors shards."""
    tensor_map = q2_index.block_tensor_names(block_idx)
    result = {}
    for tensor_name, shard_file in tensor_map.items():
        if shard_file not in shard_cache:
            shard_cache[shard_file] = mx.load(str(q2_index.shard_path(shard_file)))
        result[tensor_name] = shard_cache[shard_file][tensor_name]
    return result


# ── Phase 0: Q2 Block Micro-Benchmark ──


def run_phase_0(model_id: str):
    """Phase 0: Micro-benchmark a single Q2 block to project tok/s ceiling."""
    print("=" * 70)
    print("Phase 0: Q2 Block Micro-Benchmark (Pre-Gate)")
    print("=" * 70)

    env = get_environment_info()
    print(f"Environment: {env['chip']}, {env['memory_gb']} GB RAM")

    from mlx_lm import load
    print(f"\nLoading model: {model_id}")
    model, tokenizer = load(model_id)
    blocks = _get_model_blocks(model)
    inner = _get_inner_model(model)
    n_blocks = len(blocks)

    # Build Q4 index for reference
    model_path = _find_hf_cache_path(model_id)
    q4_index = SafetensorsBlockIndex(model_path)

    test_block_idx = 40  # Middle block
    print(f"Test block: {test_block_idx}")

    # Materialize the test block first (Q4)
    mx.eval(blocks[test_block_idx].parameters())

    # Step 1: Dequantize Q4 -> Q2 for the test block and save to temp shard
    print("\nStep 1: Quantize block to Q2...")
    q2_tensors = dequantize_block(blocks[test_block_idx], test_block_idx)
    mx.eval(list(q2_tensors.values()))
    q2_size_mb = get_q2_block_size_mb(q2_tensors)

    # Get Q4 block size for comparison
    q4_shard_cache = {}
    q4_tensors = load_block_from_safetensors(test_block_idx, q4_index, q4_shard_cache)
    mx.eval(list(q4_tensors.values()))
    q4_size_mb = sum(arr.nbytes for arr in q4_tensors.values()) / (1024 * 1024)
    del q4_shard_cache, q4_tensors

    print(f"  Q4 block size: {q4_size_mb:.1f} MB")
    print(f"  Q2 block size: {q2_size_mb:.1f} MB")
    print(f"  Reduction: {(1 - q2_size_mb / q4_size_mb) * 100:.1f}%")

    # Save Q2 block to temp safetensors
    temp_dir = Path("/tmp/h8b_phase0_q2")
    temp_dir.mkdir(parents=True, exist_ok=True)
    shard_np = {}
    save_q2_block_to_shard(q2_tensors, shard_np)
    temp_shard = temp_dir / "q2_test_block.safetensors"
    write_shard(shard_np, temp_shard)
    del shard_np, q2_tensors
    gc.collect()

    # Build a mini Q2 index for the test block
    weight_map = {}
    loaded = mx.load(str(temp_shard))
    for key in loaded.keys():
        weight_map[key] = "q2_test_block.safetensors"
    del loaded
    with open(temp_dir / "model.safetensors.index.json", "w") as f:
        json.dump({"weight_map": weight_map}, f)
    q2_test_index = Q2BlockIndex(temp_dir)

    # Evict the test block
    evict_block(blocks[test_block_idx])
    gc.collect()

    # Materialize block 0 + embeddings for hidden state generation
    mx.eval(inner.embed_tokens.parameters())
    mx.eval(blocks[0].parameters())
    dummy_ids = mx.array([[1]])
    h_input = inner.embed_tokens(dummy_ids)
    h_input = blocks[0](h_input, None, None)
    mx.eval(h_input)

    # Step 2: Load cycle benchmark
    print(f"\nStep 2: Q2 load cycle benchmark (10 cycles)...")
    process = psutil.Process()
    results = []

    for cycle in range(10):
        rss_before = process.memory_info().rss / 1e6
        metal_before = mx.get_active_memory() / 1e6

        shard_cache = {}

        t0 = time.perf_counter()
        tensors = load_q2_block(test_block_idx, q2_test_index, shard_cache)
        t_load = time.perf_counter()

        assign_q2_block_weights(blocks[test_block_idx], test_block_idx, tensors)
        t_assign = time.perf_counter()

        mx.eval(blocks[test_block_idx].parameters())
        t_eval = time.perf_counter()

        h = blocks[test_block_idx](h_input, None, None)
        mx.eval(h)
        t_forward = time.perf_counter()

        evict_block(blocks[test_block_idx])
        restore_q4_block_metadata(blocks[test_block_idx])
        t_evict = time.perf_counter()

        rss_after = process.memory_info().rss / 1e6
        metal_after = mx.get_active_memory() / 1e6
        metal_peak = mx.get_peak_memory() / 1e6

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
            "metal_active_mb": metal_after,
            "metal_peak_mb": metal_peak,
        }
        results.append(result)

        if cycle < 3 or cycle == 9:
            print(f"  Cycle {cycle}: load={result['t_load_ms']:.1f} assign={result['t_assign_ms']:.1f} "
                  f"eval={result['t_eval_ms']:.1f} fwd={result['t_forward_ms']:.1f} "
                  f"evict={result['t_evict_ms']:.1f} total={result['t_total_ms']:.1f}ms "
                  f"Metal={result['metal_active_mb']:.0f}MB")

    # Compute steady-state stats (skip warmup)
    steady = results[2:]
    p50 = {}
    for comp in ["t_load_ms", "t_assign_ms", "t_eval_ms", "t_forward_ms", "t_evict_ms", "t_total_ms"]:
        p50[comp] = float(np.percentile([r[comp] for r in steady], 50))

    print(f"\n  Steady-state p50:")
    for comp, val in p50.items():
        print(f"    {comp}: {val:.1f} ms")

    # Step 3: Multi-token residency probe
    print(f"\nStep 3: Multi-token residency probe...")

    # We need to simulate scanning all 64 streaming blocks twice
    # But we only have 1 Q2 block saved. Instead, measure file-level residency
    # after a full-file scan to estimate.
    # For now, measure residency of the test shard before/after load
    shard_path_str = str(temp_shard)
    residency_before = measure_file_residency(shard_path_str)

    # Load and eval the block (touch the pages)
    shard_cache = {}
    tensors = load_q2_block(test_block_idx, q2_test_index, shard_cache)
    assign_q2_block_weights(blocks[test_block_idx], test_block_idx, tensors)
    mx.eval(blocks[test_block_idx].parameters())
    del shard_cache, tensors
    evict_block(blocks[test_block_idx])
    restore_q4_block_metadata(blocks[test_block_idx])

    residency_after = measure_file_residency(shard_path_str)
    print(f"  Shard residency before load: {residency_before:.2%}")
    print(f"  Shard residency after load:  {residency_after:.2%}")

    # Note: real multi-token residency can only be measured with the full 64-block
    # Q2 checkpoint (Phase 1). For now, estimate based on working set math.
    n_streaming = 64
    estimated_working_set_gb = n_streaming * q2_size_mb / 1024
    estimated_page_cache_gb = env["memory_gb"] - 15.1  # rough pinned total
    if estimated_page_cache_gb < 0:
        estimated_page_cache_gb = env["available_gb"] * 0.8
    estimated_coverage = min(1.0, estimated_page_cache_gb / estimated_working_set_gb)
    print(f"  Estimated working set: {estimated_working_set_gb:.1f} GB")
    print(f"  Estimated page cache: {estimated_page_cache_gb:.1f} GB")
    print(f"  Estimated coverage: {estimated_coverage:.1%}")

    # Step 4: Gate checks
    print(f"\n{'='*70}")
    print(f"Gate Checks")
    print(f"{'='*70}")

    t_total_p50 = p50["t_total_ms"]
    t_forward_p50 = p50["t_forward_ms"]

    # Hard gate: cold-cache ceiling
    cold_ceiling = 1000 / (n_streaming * t_total_p50) if t_total_p50 > 0 else 0
    gate_hard = t_total_p50 < 150
    print(f"  Hard gate (t_total_p50 < 150ms): {'PASS' if gate_hard else 'FAIL'} ({t_total_p50:.1f}ms)")
    print(f"  Cold-cache ceiling: {cold_ceiling:.3f} tok/s")

    # Projection gate: required hit rate for 0.3 tok/s
    target_per_block = 1000 / (0.3 * n_streaming)  # ~52 ms
    if t_total_p50 > t_forward_p50:
        required_hit_rate = (t_total_p50 - target_per_block) / (t_total_p50 - t_forward_p50)
        required_hit_rate = max(0.0, min(1.0, required_hit_rate))
    else:
        required_hit_rate = 0.0
    gate_projection = required_hit_rate <= 0.80
    print(f"  Target per-block latency: {target_per_block:.1f} ms")
    print(f"  Required hit rate for 0.3 tok/s: {required_hit_rate:.2%}")
    print(f"  Projection gate (required <= 80%): {'PASS' if gate_projection else 'FAIL'}")

    # Note: multi-token residency probe is informational only at this stage
    # Full measurement requires the complete Q2 checkpoint (Phase 1)
    print(f"\n  Note: Full multi-token residency measurement requires Phase 1 checkpoint.")
    print(f"  Estimated coverage ({estimated_coverage:.0%}) suggests "
          f"{'some' if estimated_coverage > 0.3 else 'minimal'} inter-token reuse potential.")

    overall = gate_hard and gate_projection
    print(f"\n  Overall Phase 0: {'PASS' if overall else 'FAIL'}")

    log_experiment(
        experiment_name="h8b_q2_streaming",
        phase="phase_0_microbenchmark",
        config={
            "model": model_id,
            "test_block": test_block_idx,
            "n_cycles": 10,
            "q2_bits": Q2_BITS,
            "q2_group_size": Q2_GROUP_SIZE,
        },
        results={
            "q4_block_mb": round(q4_size_mb, 1),
            "q2_block_mb": round(q2_size_mb, 1),
            "size_reduction_pct": round((1 - q2_size_mb / q4_size_mb) * 100, 1),
            "component_p50": {k: round(v, 1) for k, v in p50.items()},
            "cold_ceiling_tok_s": round(cold_ceiling, 4),
            "target_per_block_ms": round(target_per_block, 1),
            "required_hit_rate_for_0.3": round(required_hit_rate, 4),
            "estimated_coverage": round(estimated_coverage, 3),
            "metal_peak_mb": round(max(r["metal_peak_mb"] for r in steady), 1),
            "gate_hard": gate_hard,
            "gate_projection": gate_projection,
            "gate_pass": overall,
        },
        env=env,
    )

    del model, tokenizer
    return overall


# ── Phase 0b: Q2 Quality Pilot ──


def run_phase_0b(model_id_7b: str):
    """Phase 0b: Quick quality check — Q2 on 3 representative blocks of 7B."""
    print("=" * 70)
    print("Phase 0b: Q2 Quality Pilot (Early Kill Switch)")
    print("=" * 70)

    env = get_environment_info()
    print(f"Model: {model_id_7b}")

    from mlx_lm import load
    model, tokenizer = load(model_id_7b)
    blocks = _get_model_blocks(model)
    inner = _get_inner_model(model)
    n_blocks = len(blocks)

    # Materialize everything
    for b in range(n_blocks):
        mx.eval(blocks[b].parameters())
    mx.eval(inner.embed_tokens.parameters())
    mx.eval(inner.norm.parameters())
    if hasattr(model, "lm_head"):
        mx.eval(model.lm_head.parameters())

    # Baseline: all-Q4 NLL on a fixed prompt
    prompt = "The key innovation of the transformer architecture is"
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    print(f"\n  Prompt: '{prompt}' ({len(tokens)} tokens)")
    print(f"  {n_blocks} blocks, testing Q2 on 3 representative blocks")

    # Generate baseline NLL (all-Q4) and capture reference tokens
    print("\n  Computing all-Q4 baseline NLL...")
    baseline_nlls, baseline_generated = _compute_generation_nll(
        model, inner, tokenizer, tokens, n_tokens=20,
    )
    baseline_mean_nll = sum(baseline_nlls) / len(baseline_nlls)
    # Extract reference tokens for teacher forcing (tokens generated by baseline)
    reference_tokens = baseline_generated[len(tokens):]
    print(f"  Baseline mean NLL: {baseline_mean_nll:.4f}")

    # Pick 3 representative blocks: early, middle, late in streaming range
    n_resident_each = max(1, n_blocks // 5)
    streaming_indices = list(range(n_resident_each, n_blocks - n_resident_each))
    test_blocks = [
        streaming_indices[0],
        streaming_indices[len(streaming_indices) // 2],
        streaming_indices[-1],
    ]
    print(f"  Test blocks (Q2): {test_blocks}")

    # Dequantize + requantize the 3 test blocks
    for bi in test_blocks:
        q2_tensors = dequantize_block(blocks[bi], bi)
        assign_q2_block_weights(blocks[bi], bi, q2_tensors)
        mx.eval(blocks[bi].parameters())
        del q2_tensors

    # Generate Q2 NLL with teacher forcing against baseline tokens
    print("  Computing 3-block-Q2 NLL (teacher-forced against baseline)...")
    q2_nlls, _ = _compute_generation_nll(
        model, inner, tokenizer, tokens, n_tokens=20,
        reference_tokens=reference_tokens,
    )
    q2_mean_nll = sum(q2_nlls) / len(q2_nlls)
    print(f"  Q2 mean NLL: {q2_mean_nll:.4f}")

    nll_delta = q2_mean_nll - baseline_mean_nll
    gate_pass = nll_delta < 1.0
    print(f"\n  NLL delta: {nll_delta:.4f}")
    print(f"  Gate (NLL delta < 1.0): {'PASS' if gate_pass else 'FAIL'}")

    if not gate_pass:
        print("  WARNING: Q2 quality is too degraded. Consider Q3 (3-bit) instead.")

    log_experiment(
        experiment_name="h8b_q2_streaming",
        phase="phase_0b_quality_pilot",
        config={
            "model": model_id_7b,
            "test_blocks": test_blocks,
            "q2_bits": Q2_BITS,
            "q2_group_size": Q2_GROUP_SIZE,
            "n_tokens": 20,
        },
        results={
            "baseline_mean_nll": round(baseline_mean_nll, 4),
            "q2_mean_nll": round(q2_mean_nll, 4),
            "nll_delta": round(nll_delta, 4),
            "per_token_baseline": [round(x, 4) for x in baseline_nlls],
            "per_token_q2": [round(x, 4) for x in q2_nlls],
            "gate_pass": gate_pass,
        },
        env=env,
    )

    del model, tokenizer
    return gate_pass


def _compute_generation_nll(model, inner, tokenizer, prompt_tokens, n_tokens=20,
                            reference_tokens=None):
    """Generate tokens and compute per-token NLL with proper KV cache.

    If reference_tokens is provided, NLL is scored against those tokens
    (teacher forcing) rather than the model's own argmax. This ensures
    a degraded model can't hide regressions by confidently predicting
    different tokens.
    """
    from mlx_lm.models.cache import make_prompt_cache

    kv_cache = make_prompt_cache(model)

    # Prefill: process all prompt tokens
    input_ids = mx.array([prompt_tokens])
    h = inner.embed_tokens(input_ids)
    mask = "causal" if h.shape[1] > 1 else None
    for i, layer in enumerate(inner.layers):
        h = layer(h, mask, kv_cache[i])
    h = inner.norm(h)
    if hasattr(model, "args") and model.args.tie_word_embeddings:
        logits = inner.embed_tokens.as_linear(h)
    else:
        logits = model.lm_head(h)
    mx.eval(logits)

    nlls = []
    tokens = list(prompt_tokens)

    for i in range(n_tokens):
        last_logits = logits[0, -1, :]
        log_probs = mx.log_softmax(last_logits)
        next_token = mx.argmax(last_logits).item()

        if i > 0:
            # Score against reference if provided, else against own prediction
            score_token = reference_tokens[i] if reference_tokens else next_token
            nll = -log_probs[score_token].item()
            nlls.append(nll)

        # Use reference token for next step if in teacher-forcing mode
        step_token = reference_tokens[i] if reference_tokens else next_token
        tokens.append(step_token)

        # Decode step with KV cache
        next_ids = mx.array([[step_token]])
        h = inner.embed_tokens(next_ids)
        for j, layer in enumerate(inner.layers):
            h = layer(h, None, kv_cache[j])
        h = inner.norm(h)
        if hasattr(model, "args") and model.args.tie_word_embeddings:
            logits = inner.embed_tokens.as_linear(h)
        else:
            logits = model.lm_head(h)
        mx.eval(logits)

    return nlls, tokens


# ── Phase 1: Q2 Checkpoint Preparation ──


def run_phase_1(model_id: str):
    """Phase 1: Quantize 64 streaming blocks from Q4 to Q2 safetensors."""
    print("=" * 70)
    print("Phase 1: Q2 Checkpoint Preparation")
    print("=" * 70)

    env = get_environment_info()
    print(f"Model: {model_id}")

    from mlx_lm import load
    model, tokenizer = load(model_id)
    blocks = _get_model_blocks(model)
    inner = _get_inner_model(model)
    n_blocks = len(blocks)

    n_first = 8
    n_last = 8
    streaming_indices = list(range(n_first, n_blocks - n_last))
    print(f"  Total blocks: {n_blocks}")
    print(f"  Streaming blocks to quantize: {len(streaming_indices)} (indices {n_first}-{n_blocks - n_last - 1})")

    Q2_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Output dir: {Q2_CACHE_DIR}")

    # Process blocks in shard-sized groups
    weight_map = {}
    shard_idx = 0
    current_shard = {}
    blocks_in_shard = 0
    total_q2_bytes = 0

    for i, bi in enumerate(streaming_indices):
        print(f"\n  Block {bi} ({i+1}/{len(streaming_indices)})...")

        # Materialize Q4 block
        mx.eval(blocks[bi].parameters())

        # Dequantize Q4 -> Q2
        q2_tensors = dequantize_block(blocks[bi], bi)
        mx.eval(list(q2_tensors.values()))

        block_bytes = sum(arr.nbytes for arr in q2_tensors.values())
        total_q2_bytes += block_bytes
        print(f"    Q2 size: {block_bytes / (1024*1024):.1f} MB")

        # Add to current shard
        save_q2_block_to_shard(q2_tensors, current_shard)
        shard_file = f"q2_shard_{shard_idx:02d}.safetensors"
        for key in q2_tensors:
            weight_map[key] = shard_file
        blocks_in_shard += 1

        # Evict the Q4 block
        del q2_tensors
        evict_block(blocks[bi])
        gc.collect()

        # Write shard if full
        if blocks_in_shard >= BLOCKS_PER_SHARD or i == len(streaming_indices) - 1:
            shard_path = Q2_CACHE_DIR / shard_file
            print(f"    Writing shard {shard_idx}: {len(current_shard)} tensors, "
                  f"{sum(v.nbytes for v in current_shard.values()) / (1024**3):.2f} GB")
            write_shard(current_shard, shard_path)
            current_shard = {}
            blocks_in_shard = 0
            shard_idx += 1

        avail = get_available_memory_gb()
        rss = get_rss_mb()
        print(f"    RSS={rss:.0f} MB, Avail={avail:.1f} GB")

    # Write index
    index_data = {"weight_map": weight_map}
    with open(Q2_CACHE_DIR / "model.safetensors.index.json", "w") as f:
        json.dump(index_data, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Phase 1 Complete")
    print(f"{'='*70}")
    print(f"  Total Q2 size: {total_q2_bytes / (1024**3):.2f} GB")
    print(f"  Avg block: {total_q2_bytes / len(streaming_indices) / (1024*1024):.1f} MB")
    print(f"  Shards written: {shard_idx}")
    print(f"  Index: {Q2_CACHE_DIR / 'model.safetensors.index.json'}")

    # Verify: load one block back
    print(f"\n  Verification: loading block {streaming_indices[0]} from Q2 shards...")
    q2_index = Q2BlockIndex(Q2_CACHE_DIR)
    shard_cache = {}
    test_tensors = load_q2_block(streaming_indices[0], q2_index, shard_cache)
    assign_q2_block_weights(blocks[streaming_indices[0]], streaming_indices[0], test_tensors)
    mx.eval(blocks[streaming_indices[0]].parameters())

    # Forward pass check
    dummy_ids = mx.array([[1]])
    mx.eval(inner.embed_tokens.parameters())
    h = inner.embed_tokens(dummy_ids)
    h = blocks[streaming_indices[0]](h, None, None)
    mx.eval(h)
    has_nan = bool(mx.any(mx.isnan(h)).item())
    print(f"  Forward pass: {'FAIL (NaN)' if has_nan else 'OK'}")

    evict_block(blocks[streaming_indices[0]])
    restore_q4_block_metadata(blocks[streaming_indices[0]])
    del shard_cache, test_tensors

    gate_pass = not has_nan
    print(f"  Gate: {'PASS' if gate_pass else 'FAIL'}")

    log_experiment(
        experiment_name="h8b_q2_streaming",
        phase="phase_1_checkpoint_prep",
        config={
            "model": model_id,
            "n_streaming": len(streaming_indices),
            "q2_bits": Q2_BITS,
            "q2_group_size": Q2_GROUP_SIZE,
            "blocks_per_shard": BLOCKS_PER_SHARD,
        },
        results={
            "total_q2_gb": round(total_q2_bytes / (1024**3), 2),
            "avg_block_mb": round(total_q2_bytes / len(streaming_indices) / (1024*1024), 1),
            "n_shards": shard_idx,
            "verification_pass": not has_nan,
            "gate_pass": gate_pass,
        },
        env=env,
    )

    del model, tokenizer
    return gate_pass


# ── Phase 2: Mixed-Precision Streaming Integration ──


def run_phase_2(model_id: str, n_tokens: int = 20):
    """Phase 2: 8+8 Q4 resident + 64 Q2 streamed, generate tokens."""
    print("=" * 70)
    print("Phase 2: Mixed-Precision Streaming Integration")
    print("=" * 70)

    env = get_environment_info()
    available_idle = get_available_memory_gb()
    print(f"Environment: {env['chip']}, {env['memory_gb']} GB RAM, {available_idle:.1f} GB available")
    print(f"Model: {model_id}")

    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    print(f"\nStep 1: Loading model...")
    t_load_start = time.perf_counter()
    model, tokenizer = load(model_id)
    print(f"  Loaded in {time.perf_counter() - t_load_start:.0f}s")

    blocks = _get_model_blocks(model)
    inner = _get_inner_model(model)
    n_blocks = len(blocks)

    # Load Q2 index
    q2_index = Q2BlockIndex(Q2_CACHE_DIR)

    n_first = 8
    n_last = 8
    resident_indices = list(range(n_first)) + list(range(n_blocks - n_last, n_blocks))
    resident_set = set(resident_indices)
    streaming_indices = [i for i in range(n_blocks) if i not in resident_set]

    print(f"  Config: 8+8 Q4 resident / {len(streaming_indices)} Q2 streamed")

    # Step 2: Incremental setup
    print(f"\nStep 2: Setting up blocks...")
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

    mx.eval(inner.embed_tokens.parameters())
    mx.eval(inner.norm.parameters())
    if hasattr(model, "lm_head"):
        mx.eval(model.lm_head.parameters())

    setup_s = time.perf_counter() - t_setup
    avail_after_setup = get_available_memory_gb()
    rss_after_setup = get_rss_mb()
    metal_after_setup = mx.get_active_memory() / 1e6
    print(f"  Setup done in {setup_s:.0f}s")
    print(f"  RSS={rss_after_setup:.0f} MB, Avail={avail_after_setup:.1f} GB, Metal={metal_after_setup:.0f} MB")
    mx.clear_cache()

    # Step 3: Generate tokens
    print(f"\nStep 3: Generating {n_tokens} tokens...")

    prompt = "The key innovation of the transformer architecture is"
    tokens = tokenizer.encode(prompt)
    print(f"  Prompt: '{prompt}' ({len(tokens)} tokens)")

    kv_cache = make_prompt_cache(model)

    # Prefill
    t_prefill_start = time.perf_counter()
    input_ids = mx.array([tokens])
    h = inner.embed_tokens(input_ids)
    mask = "causal"

    for i, layer in enumerate(inner.layers):
        if i in resident_set:
            h = layer(h, mask, kv_cache[i])
        else:
            # Per-block shard scope: load, extract, release immediately
            # This ensures only one block's worth of memory is pinned at a time
            shard_cache = {}
            q2_tensors = load_q2_block(i, q2_index, shard_cache)
            assign_q2_block_weights(blocks[i], i, q2_tensors)
            mx.eval(blocks[i].parameters())
            del shard_cache, q2_tensors

            h = layer(h, mask, kv_cache[i])
            mx.eval(h, kv_cache[i].state)

            evict_block(blocks[i])
            restore_q4_block_metadata(blocks[i])

    h = inner.norm(h)
    if hasattr(model, "args") and model.args.tie_word_embeddings:
        logits = inner.embed_tokens.as_linear(h)
    else:
        logits = model.lm_head(h)
    mx.eval(logits)

    prefill_logits = logits[0, -1, :]
    next_token = mx.argmax(prefill_logits).item()
    tokens.append(next_token)

    ttft_ms = (time.perf_counter() - t_prefill_start) * 1000
    print(f"  Prefill: {ttft_ms:.0f} ms, first token: '{tokenizer.decode([next_token])}'")

    # Decode
    vm_before = get_vm_stat()
    per_token_ms = []
    all_load_ms = []
    avail_history = []
    rss_history = []
    nlls = []

    for tok_i in range(n_tokens - 1):
        t0 = time.perf_counter()
        input_ids = mx.array([[tokens[-1]]])
        h = inner.embed_tokens(input_ids)

        token_load_ms = []
        for i, layer in enumerate(inner.layers):
            if i in resident_set:
                h = layer(h, None, kv_cache[i])
            else:
                t_block = time.perf_counter()
                # Per-block shard scope to avoid holding refs to all shards
                shard_cache = {}
                q2_tensors = load_q2_block(i, q2_index, shard_cache)
                assign_q2_block_weights(blocks[i], i, q2_tensors)
                mx.eval(blocks[i].parameters())
                del shard_cache, q2_tensors
                block_load_ms = (time.perf_counter() - t_block) * 1000
                token_load_ms.append(block_load_ms)

                h = layer(h, None, kv_cache[i])
                mx.eval(h, kv_cache[i].state)

                evict_block(blocks[i])
                restore_q4_block_metadata(blocks[i])

        h = inner.norm(h)
        if hasattr(model, "args") and model.args.tie_word_embeddings:
            logits = inner.embed_tokens.as_linear(h)
        else:
            logits = model.lm_head(h)

        # NLL for directional quality tracking (self-score only — primary quality
        # gate is Phase 4 which uses teacher forcing against a reference)
        log_probs = mx.log_softmax(logits[0, -1, :])
        next_token = mx.argmax(logits[0, -1, :]).item()
        nll = -log_probs[next_token].item()
        nlls.append(nll)
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
                  f"RSS={rss_now:.0f} MB, Avail={avail_now:.1f} GB, NLL={nll:.2f}, '{decoded}'")

    vm_after = get_vm_stat()
    vm_delta = vm_stat_delta(vm_before, vm_after)
    peak_rss = get_peak_rss_mb()
    metal_peak = mx.get_peak_memory() / 1e6

    generated_text = tokenizer.decode(tokens[len(tokenizer.encode(prompt)):])
    print(f"\n  Generated: '{generated_text[:200]}'")

    # Statistics
    steady = per_token_ms[2:] if len(per_token_ms) > 2 else per_token_ms
    avg_tok_ms = sum(steady) / len(steady) if steady else 0
    tok_s = 1000 / avg_tok_ms if avg_tok_ms > 0 else 0
    min_avail = min(avail_history) if avail_history else 0
    max_rss = max(rss_history) if rss_history else 0
    rss_variance = max(rss_history) - min(rss_history) if rss_history else 0
    mean_nll = sum(nlls) / len(nlls) if nlls else 0

    print(f"\n{'='*70}")
    print(f"RESULTS — Phase 2: Mixed-Precision Streaming")
    print(f"{'='*70}")
    print(f"  tok/s: {tok_s:.4f} ({avg_tok_ms:.0f} ms/tok)")
    print(f"  TTFT: {ttft_ms:.0f} ms")
    print(f"  Peak RSS: {peak_rss:.0f} MB")
    print(f"  Metal peak: {metal_peak:.0f} MB")
    print(f"  Min available: {min_avail:.1f} GB")
    print(f"  RSS variance: {rss_variance:.0f} MB")
    print(f"  Mean NLL: {mean_nll:.4f}")

    if all_load_ms:
        load_p50 = float(np.percentile(all_load_ms, 50))
        load_p95 = float(np.percentile(all_load_ms, 95))
        print(f"  Block load p50/p95: {load_p50:.0f}/{load_p95:.0f} ms")

    print(f"  Pageouts: {vm_delta['pageout_delta_mb']:.0f} MB")
    print(f"  Pageins: {vm_delta['pagein_delta_mb']:.0f} MB")

    # Coherence check via NLL threshold (not subjective)
    words = generated_text.split()
    if len(words) >= 4:
        repeated = sum(1 for j in range(1, len(words)) if words[j] == words[j-1])
        gate_coherent = repeated / len(words) < 0.5 and mean_nll < 5.0
    else:
        gate_coherent = mean_nll < 5.0

    gate_tok = tok_s >= 0.1  # Minimum viable (cold-cache adjusted)
    gate_rss = rss_variance < 500
    gate_avail = min_avail > 2.0

    print(f"\n  --- Gate Checks ---")
    print(f"    tok/s >= 0.1 (minimum): {'PASS' if gate_tok else 'FAIL'} ({tok_s:.4f})")
    print(f"    Coherent (NLL < 5.0, no repeat): {'PASS' if gate_coherent else 'FAIL'}")
    print(f"    RSS variance < 500 MB: {'PASS' if gate_rss else 'FAIL'} ({rss_variance:.0f} MB)")
    print(f"    Available > 2 GB: {'PASS' if gate_avail else 'FAIL'} ({min_avail:.1f} GB)")
    overall = gate_tok and gate_coherent and gate_rss and gate_avail
    print(f"    Overall: {'PASS' if overall else 'FAIL'}")

    log_experiment(
        experiment_name="h8b_q2_streaming",
        phase="phase_2_mixed_precision",
        config={
            "model": model_id,
            "n_tokens": n_tokens,
            "n_blocks": n_blocks,
            "n_resident": len(resident_indices),
            "n_streaming": len(streaming_indices),
            "config_name": "8+8 Q4 / 64 Q2",
            "q2_bits": Q2_BITS,
            "q2_group_size": Q2_GROUP_SIZE,
        },
        results={
            "tok_s": round(tok_s, 4),
            "avg_tok_ms": round(avg_tok_ms, 1),
            "ttft_ms": round(ttft_ms, 1),
            "peak_rss_mb": round(peak_rss, 1),
            "metal_peak_mb": round(metal_peak, 1),
            "min_available_gb": round(min_avail, 2),
            "rss_variance_mb": round(rss_variance, 1),
            "mean_nll": round(mean_nll, 4),
            "block_load_p50_ms": round(float(np.percentile(all_load_ms, 50)), 1) if all_load_ms else 0,
            "block_load_p95_ms": round(float(np.percentile(all_load_ms, 95)), 1) if all_load_ms else 0,
            "pageout_mb": vm_delta["pageout_delta_mb"],
            "pagein_mb": vm_delta["pagein_delta_mb"],
            "available_gb_idle": round(available_idle, 2),
            "generated_text_preview": generated_text[:200],
            "gate_pass": overall,
        },
        env=env,
    )

    del model, tokenizer
    return overall


# ── Phase 3: Cache Optimization ──


def _synthetic_madvise_bench(file_path: str, n_runs: int = 3) -> dict:
    """Benchmark sequential read with vs without madvise(MADV_SEQUENTIAL).

    Returns dict with throughput_baseline_gbs, throughput_madvise_gbs, improvement_pct.
    """
    MADV_SEQUENTIAL = 2  # macOS
    file_size = os.path.getsize(file_path)
    buf_size = 1024 * 1024  # 1 MB read chunks

    results = {"baseline": [], "madvise": []}

    for label in ["baseline", "madvise"]:
        for _ in range(n_runs):
            # Drop caches by reading a different pattern first
            gc.collect()

            fd = os.open(file_path, os.O_RDONLY)
            addr = libc.mmap(
                None, ctypes.c_size_t(file_size), ctypes.c_int(PROT_READ),
                ctypes.c_int(MAP_PRIVATE), ctypes.c_int(fd), ctypes.c_long(0),
            )
            if addr == ctypes.c_void_p(-1).value:
                os.close(fd)
                continue

            if label == "madvise":
                libc.madvise(ctypes.c_void_p(addr), ctypes.c_size_t(file_size),
                             ctypes.c_int(MADV_SEQUENTIAL))

            # Sequential read: touch every page
            t0 = time.perf_counter()
            offset = 0
            while offset < file_size:
                chunk = min(buf_size, file_size - offset)
                # Read by accessing memory (force page faults)
                ctypes.memmove(
                    ctypes.create_string_buffer(chunk),
                    ctypes.c_void_p(addr + offset),
                    chunk,
                )
                offset += chunk
            elapsed = time.perf_counter() - t0

            libc.munmap(ctypes.c_void_p(addr), ctypes.c_size_t(file_size))
            os.close(fd)

            throughput = (file_size / (1024**3)) / elapsed if elapsed > 0 else 0
            results[label].append(throughput)

    baseline_gbs = float(np.median(results["baseline"])) if results["baseline"] else 0
    madvise_gbs = float(np.median(results["madvise"])) if results["madvise"] else 0
    improvement = ((madvise_gbs - baseline_gbs) / baseline_gbs * 100) if baseline_gbs > 0 else 0

    return {
        "throughput_baseline_gbs": round(baseline_gbs, 2),
        "throughput_madvise_gbs": round(madvise_gbs, 2),
        "improvement_pct": round(improvement, 1),
    }


def _prefault_shard_range(file_path: str, offset: int, length: int):
    """Pre-fault a byte range into page cache via a sidecar mmap read."""
    fd = os.open(file_path, os.O_RDONLY)
    try:
        addr = libc.mmap(
            None, ctypes.c_size_t(length), ctypes.c_int(PROT_READ),
            ctypes.c_int(MAP_PRIVATE), ctypes.c_int(fd), ctypes.c_long(offset),
        )
        if addr == ctypes.c_void_p(-1).value:
            return
        try:
            # Touch every page to bring it into cache
            page_step = PAGE_SIZE * 16  # stride by 16 pages for speed
            pos = 0
            dummy = ctypes.c_char()
            while pos < length:
                ctypes.memmove(ctypes.byref(dummy), ctypes.c_void_p(addr + pos), 1)
                pos += page_step
        finally:
            libc.munmap(ctypes.c_void_p(addr), ctypes.c_size_t(length))
    finally:
        os.close(fd)


def _readahead_next_block(block_idx: int, q2_index: Q2BlockIndex):
    """Issue pread() calls on the next block's shard ranges in a background thread."""
    if not q2_index.has_block(block_idx):
        return
    for shard_file in q2_index.block_shards(block_idx):
        shard_path = str(q2_index.shard_path(shard_file))
        if os.path.exists(shard_path):
            file_size = os.path.getsize(shard_path)
            # Read first 64KB to prime the readahead
            fd = os.open(shard_path, os.O_RDONLY)
            try:
                os.pread(fd, 65536, 0)
            finally:
                os.close(fd)


def _measure_block_residency_before_load(block_idx: int, q2_index: Q2BlockIndex) -> float:
    """Measure page cache residency for a block's shard files before loading."""
    residencies = []
    for shard_file in q2_index.block_shards(block_idx):
        shard_path = str(q2_index.shard_path(shard_file))
        if os.path.exists(shard_path):
            r = measure_file_residency(shard_path)
            residencies.append(r)
    return float(np.mean(residencies)) if residencies else 0.0


def _run_streaming_pass(
    model, inner, blocks, kv_cache, q2_index, resident_set,
    tokens, n_tokens, tokenizer,
    optimization: str = "none",
    measure_residency: bool = False,
):
    """Run a streaming decode pass with optional optimization.

    optimization: "none", "prefault", or "readahead"
    Returns dict with per-token metrics.
    """
    from mlx_lm.models.cache import make_prompt_cache

    n_blocks = len(blocks)
    streaming_indices = [i for i in range(n_blocks) if i not in resident_set]

    # Fresh KV cache for each pass
    kv_cache = make_prompt_cache(model)

    # Prefill
    t_prefill = time.perf_counter()
    input_ids = mx.array([tokens])
    h = inner.embed_tokens(input_ids)
    mask = "causal"

    for i, layer in enumerate(inner.layers):
        if i in resident_set:
            h = layer(h, mask, kv_cache[i])
        else:
            shard_cache = {}
            q2_tensors = load_q2_block(i, q2_index, shard_cache)
            assign_q2_block_weights(blocks[i], i, q2_tensors)
            mx.eval(blocks[i].parameters())
            del shard_cache, q2_tensors

            h = layer(h, mask, kv_cache[i])
            mx.eval(h, kv_cache[i].state)

            evict_block(blocks[i])
            restore_q4_block_metadata(blocks[i])

    h = inner.norm(h)
    if hasattr(model, "args") and model.args.tie_word_embeddings:
        logits = inner.embed_tokens.as_linear(h)
    else:
        logits = model.lm_head(h)
    mx.eval(logits)

    next_token = mx.argmax(logits[0, -1, :]).item()
    generated = list(tokens) + [next_token]
    ttft_ms = (time.perf_counter() - t_prefill) * 1000

    # Decode
    per_token_ms = []
    all_load_ms = []
    residency_per_token = []

    for tok_i in range(n_tokens - 1):
        t0 = time.perf_counter()
        input_ids = mx.array([[generated[-1]]])
        h = inner.embed_tokens(input_ids)

        token_load_ms = []
        token_residencies = []

        for i, layer in enumerate(inner.layers):
            if i in resident_set:
                h = layer(h, None, kv_cache[i])
            else:
                # Measure residency before load
                if measure_residency:
                    r = _measure_block_residency_before_load(i, q2_index)
                    token_residencies.append(r)

                # Apply optimization
                if optimization == "prefault":
                    for shard_file in q2_index.block_shards(i):
                        sp = str(q2_index.shard_path(shard_file))
                        if os.path.exists(sp):
                            _prefault_shard_range(sp, 0, os.path.getsize(sp))
                elif optimization == "readahead":
                    # Readahead the NEXT streaming block in a background thread
                    next_streaming = None
                    for si in streaming_indices:
                        if si > i:
                            next_streaming = si
                            break
                    if next_streaming is not None:
                        t = threading.Thread(
                            target=_readahead_next_block,
                            args=(next_streaming, q2_index),
                        )
                        t.start()

                t_block = time.perf_counter()
                shard_cache = {}
                q2_tensors = load_q2_block(i, q2_index, shard_cache)
                assign_q2_block_weights(blocks[i], i, q2_tensors)
                mx.eval(blocks[i].parameters())
                del shard_cache, q2_tensors
                block_load_ms = (time.perf_counter() - t_block) * 1000
                token_load_ms.append(block_load_ms)

                h = layer(h, None, kv_cache[i])
                mx.eval(h, kv_cache[i].state)

                evict_block(blocks[i])
                restore_q4_block_metadata(blocks[i])

        h = inner.norm(h)
        if hasattr(model, "args") and model.args.tie_word_embeddings:
            logits = inner.embed_tokens.as_linear(h)
        else:
            logits = model.lm_head(h)
        mx.eval(logits)

        next_token = mx.argmax(logits[0, -1, :]).item()
        generated.append(next_token)

        tok_ms = (time.perf_counter() - t0) * 1000
        per_token_ms.append(tok_ms)
        all_load_ms.extend(token_load_ms)
        if token_residencies:
            residency_per_token.append(float(np.mean(token_residencies)))

    steady = per_token_ms[2:] if len(per_token_ms) > 2 else per_token_ms
    avg_ms = sum(steady) / len(steady) if steady else 0
    tok_s = 1000 / avg_ms if avg_ms > 0 else 0

    return {
        "tok_s": tok_s,
        "avg_tok_ms": avg_ms,
        "ttft_ms": ttft_ms,
        "per_token_ms": per_token_ms,
        "block_load_p50_ms": float(np.percentile(all_load_ms, 50)) if all_load_ms else 0,
        "block_load_p95_ms": float(np.percentile(all_load_ms, 95)) if all_load_ms else 0,
        "mean_block_residency": float(np.mean(residency_per_token)) if residency_per_token else None,
        "generated_tokens": generated,
    }


def run_phase_3(model_id: str, n_tokens: int = 20):
    """Phase 3: Cache optimization — A/B comparison of madvise/readahead."""
    print("=" * 70)
    print("Phase 3: Cache Optimization")
    print("=" * 70)

    env = get_environment_info()
    print(f"Environment: {env['chip']}, {env['memory_gb']} GB RAM")
    print(f"Model: {model_id}")

    # Step 1: Synthetic madvise benchmark
    print(f"\nStep 1: Synthetic madvise benchmark...")
    # Find a Q2 shard to use as test file
    q2_index = Q2BlockIndex(Q2_CACHE_DIR)
    # Pick first shard
    test_shard = None
    for shard_file in sorted(os.listdir(Q2_CACHE_DIR)):
        if shard_file.endswith(".safetensors"):
            test_shard = str(Q2_CACHE_DIR / shard_file)
            break

    madvise_skip = False
    madvise_result = None
    if test_shard:
        file_mb = os.path.getsize(test_shard) / (1024**2)
        print(f"  Test file: {test_shard} ({file_mb:.0f} MB)")
        madvise_result = _synthetic_madvise_bench(test_shard)
        print(f"  Baseline throughput: {madvise_result['throughput_baseline_gbs']:.2f} GB/s")
        print(f"  madvise throughput:  {madvise_result['throughput_madvise_gbs']:.2f} GB/s")
        print(f"  Improvement: {madvise_result['improvement_pct']:.1f}%")
        if madvise_result['improvement_pct'] < 5.0:
            print(f"  madvise shows no meaningful improvement — skipping Optimization A")
            madvise_skip = True
    else:
        print("  No Q2 shards found — run Phase 1 first")
        return False

    # Step 2: Load model and run A/B comparisons
    print(f"\nStep 2: Loading model for A/B comparison...")
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    model, tokenizer = load(model_id)
    blocks = _get_model_blocks(model)
    inner = _get_inner_model(model)
    n_blocks = len(blocks)

    n_first = 8
    n_last = 8
    resident_indices = list(range(n_first)) + list(range(n_blocks - n_last, n_blocks))
    resident_set = set(resident_indices)

    # Setup: materialize and evict
    for idx in range(n_blocks):
        mx.eval(blocks[idx].parameters())
        if idx not in resident_set:
            evict_block(blocks[idx])
            gc.collect()

    mx.eval(inner.embed_tokens.parameters())
    mx.eval(inner.norm.parameters())
    if hasattr(model, "lm_head"):
        mx.eval(model.lm_head.parameters())

    prompt = "The key innovation of the transformer architecture is"
    tokens = tokenizer.encode(prompt)
    kv_cache = make_prompt_cache(model)

    configs = ["none"]
    if not madvise_skip:
        configs.append("prefault")
    configs.append("readahead")

    results_by_config = {}

    for config_name in configs:
        print(f"\n  Running config: {config_name} ({n_tokens} tokens)...")
        vm_before = get_vm_stat()

        pass_result = _run_streaming_pass(
            model, inner, blocks, kv_cache, q2_index, resident_set,
            tokens, n_tokens, tokenizer,
            optimization=config_name,
            measure_residency=True,
        )

        vm_after = get_vm_stat()
        vm_d = vm_stat_delta(vm_before, vm_after)

        text = tokenizer.decode(pass_result["generated_tokens"][len(tokens):])
        print(f"    tok/s: {pass_result['tok_s']:.4f} ({pass_result['avg_tok_ms']:.0f} ms/tok)")
        print(f"    TTFT: {pass_result['ttft_ms']:.0f} ms")
        print(f"    Block load p50/p95: {pass_result['block_load_p50_ms']:.0f}/{pass_result['block_load_p95_ms']:.0f} ms")
        if pass_result["mean_block_residency"] is not None:
            print(f"    Mean block residency: {pass_result['mean_block_residency']:.2%}")
        print(f"    Pageins: {vm_d['pagein_delta_mb']:.0f} MB, Pageouts: {vm_d['pageout_delta_mb']:.0f} MB")
        print(f"    Generated: '{text[:100]}...'")

        results_by_config[config_name] = {
            "tok_s": round(pass_result["tok_s"], 4),
            "avg_tok_ms": round(pass_result["avg_tok_ms"], 1),
            "ttft_ms": round(pass_result["ttft_ms"], 1),
            "block_load_p50_ms": round(pass_result["block_load_p50_ms"], 1),
            "block_load_p95_ms": round(pass_result["block_load_p95_ms"], 1),
            "mean_block_residency": round(pass_result["mean_block_residency"], 4) if pass_result["mean_block_residency"] is not None else None,
            "pagein_mb": vm_d["pagein_delta_mb"],
            "pageout_mb": vm_d["pageout_delta_mb"],
        }

    # Step 3: Compare
    print(f"\n{'='*70}")
    print(f"RESULTS — Phase 3: Cache Optimization A/B")
    print(f"{'='*70}")

    baseline_tok_s = results_by_config["none"]["tok_s"]
    print(f"  Baseline (none): {baseline_tok_s:.4f} tok/s")
    best_config = "none"
    best_tok_s = baseline_tok_s

    for config_name in configs:
        if config_name == "none":
            continue
        tok_s = results_by_config[config_name]["tok_s"]
        improvement = ((tok_s - baseline_tok_s) / baseline_tok_s * 100) if baseline_tok_s > 0 else 0
        print(f"  {config_name}: {tok_s:.4f} tok/s ({improvement:+.1f}%)")
        if tok_s > best_tok_s:
            best_tok_s = tok_s
            best_config = config_name

    overall_improvement = ((best_tok_s - baseline_tok_s) / baseline_tok_s * 100) if baseline_tok_s > 0 else 0
    gate_pass = overall_improvement >= 10.0
    print(f"\n  Best config: {best_config} ({overall_improvement:+.1f}% vs baseline)")
    print(f"  Gate (>= 10% improvement): {'PASS' if gate_pass else 'FAIL'}")

    if not gate_pass:
        print("  Note: No optimization meaningfully improved tok/s.")
        print("  macOS may already optimize NVMe sequential reads sufficiently.")
        print("  For lower-level control, see issue #17 (C/Metal staging buffers).")

    log_experiment(
        experiment_name="h8b_q2_streaming",
        phase="phase_3_cache_optimization",
        config={
            "model": model_id,
            "n_tokens": n_tokens,
            "configs_tested": configs,
        },
        results={
            "madvise_synthetic": madvise_result,
            "madvise_skipped": madvise_skip,
            "per_config": results_by_config,
            "best_config": best_config,
            "best_tok_s": round(best_tok_s, 4),
            "baseline_tok_s": round(baseline_tok_s, 4),
            "improvement_pct": round(overall_improvement, 1),
            "gate_pass": gate_pass,
        },
        env=env,
    )

    del model, tokenizer
    return gate_pass


# ── Phase 4: Quality Validation ──


def _compute_ppl_on_text(model, inner, tokenizer, text: str, max_tokens: int = 200) -> float:
    """Compute perplexity on a text sample. Returns PPL (exp of mean NLL)."""
    tokens = tokenizer.encode(text)[:max_tokens]
    if len(tokens) < 2:
        return float("inf")

    input_ids = mx.array([tokens])
    h = inner.embed_tokens(input_ids)
    mask = "causal"
    for i, layer in enumerate(inner.layers):
        h = layer(h, mask, None)
    h = inner.norm(h)
    if hasattr(model, "args") and model.args.tie_word_embeddings:
        logits = inner.embed_tokens.as_linear(h)
    else:
        logits = model.lm_head(h)
    mx.eval(logits)

    # NLL for each token (teacher-forced: score token t+1 given tokens 0..t)
    log_probs = mx.log_softmax(logits[0, :-1, :], axis=-1)
    target_tokens = mx.array(tokens[1:])
    per_token_nll = -log_probs[mx.arange(len(tokens) - 1), target_tokens]
    mx.eval(per_token_nll)
    mean_nll = float(mx.mean(per_token_nll).item())
    return float(np.exp(mean_nll))


def run_phase_4(model_id_7b: str, model_id_72b: str, n_tokens: int = 20):
    """Phase 4: Quality validation — PPL comparison on 7B + 72B sanity check."""
    print("=" * 70)
    print("Phase 4: Quality Validation")
    print("=" * 70)

    env = get_environment_info()

    # ── Phase 4a: 7B PPL comparison ──
    print(f"\n{'='*70}")
    print(f"Phase 4a: 7B PPL Comparison")
    print(f"{'='*70}")

    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    print(f"  Loading 7B model: {model_id_7b}")
    model_7b, tokenizer_7b = load(model_id_7b)
    blocks_7b = _get_model_blocks(model_7b)
    inner_7b = _get_inner_model(model_7b)
    n_blocks_7b = len(blocks_7b)

    # Materialize all blocks
    for b in range(n_blocks_7b):
        mx.eval(blocks_7b[b].parameters())
    mx.eval(inner_7b.embed_tokens.parameters())
    mx.eval(inner_7b.norm.parameters())
    if hasattr(model_7b, "lm_head"):
        mx.eval(model_7b.lm_head.parameters())

    # Calibration text
    cal_text = (
        "The transformer architecture, introduced in the seminal paper 'Attention is All You Need', "
        "revolutionized natural language processing by replacing recurrent neural networks with "
        "self-attention mechanisms. This innovation enabled parallel processing of input sequences, "
        "dramatically reducing training time while improving performance on a wide range of tasks "
        "including machine translation, text summarization, and question answering. The key insight "
        "was that attention alone, without recurrence or convolution, could capture long-range "
        "dependencies in text more effectively than previous approaches. Subsequent work built on "
        "this foundation with models like BERT, GPT, and T5, each pushing the boundaries of what "
        "was possible with language models."
    )

    # Baseline: all-Q4 PPL
    print(f"\n  Computing all-Q4 baseline PPL...")
    ppl_q4 = _compute_ppl_on_text(model_7b, inner_7b, tokenizer_7b, cal_text)
    print(f"  All-Q4 PPL: {ppl_q4:.4f}")

    # Mixed Q4/Q2: proportional to 72B config (8+8 resident, rest Q2)
    n_resident_each = max(1, n_blocks_7b // 10)  # ~10% each side
    streaming_7b = list(range(n_resident_each, n_blocks_7b - n_resident_each))
    print(f"  Converting {len(streaming_7b)} blocks to Q2 (indices {streaming_7b[0]}-{streaming_7b[-1]})...")

    for bi in streaming_7b:
        q2_tensors = dequantize_block(blocks_7b[bi], bi)
        assign_q2_block_weights(blocks_7b[bi], bi, q2_tensors)
        mx.eval(blocks_7b[bi].parameters())
        del q2_tensors

    ppl_q2 = _compute_ppl_on_text(model_7b, inner_7b, tokenizer_7b, cal_text)
    print(f"  Mixed Q4/Q2 PPL: {ppl_q2:.4f}")

    ppl_delta_4a = ppl_q2 - ppl_q4
    gate_4a = ppl_delta_4a < 1.0
    print(f"  PPL delta: {ppl_delta_4a:.4f}")
    print(f"  Gate (PPL delta < 1.0): {'PASS' if gate_4a else 'FAIL'}")

    # NLL stability over 50 tokens
    print(f"\n  NLL stability over 50 tokens...")
    prompt_text = "The key innovation of the transformer architecture is"
    prompt_tokens = tokenizer_7b.encode(prompt_text)
    q2_nlls_50, _ = _compute_generation_nll(
        model_7b, inner_7b, tokenizer_7b, prompt_tokens, n_tokens=50,
    )
    nll_std = float(np.std(q2_nlls_50)) if q2_nlls_50 else 0
    nll_mean = float(np.mean(q2_nlls_50)) if q2_nlls_50 else 0
    print(f"  Mean NLL over 50 tokens: {nll_mean:.4f} (std: {nll_std:.4f})")

    del model_7b, tokenizer_7b
    gc.collect()

    # ── Phase 4b: 72B quality sanity check ──
    print(f"\n{'='*70}")
    print(f"Phase 4b: 72B Quality Sanity Check")
    print(f"{'='*70}")

    print(f"  Loading 72B model: {model_id_72b}")
    model_72b, tokenizer_72b = load(model_id_72b)
    blocks_72b = _get_model_blocks(model_72b)
    inner_72b = _get_inner_model(model_72b)
    n_blocks_72b = len(blocks_72b)

    n_first = 8
    n_last = 8
    resident_indices = list(range(n_first)) + list(range(n_blocks_72b - n_last, n_blocks_72b))
    resident_set = set(resident_indices)

    # Setup blocks
    for idx in range(n_blocks_72b):
        mx.eval(blocks_72b[idx].parameters())
        if idx not in resident_set:
            evict_block(blocks_72b[idx])
            gc.collect()

    mx.eval(inner_72b.embed_tokens.parameters())
    mx.eval(inner_72b.norm.parameters())
    if hasattr(model_72b, "lm_head"):
        mx.eval(model_72b.lm_head.parameters())

    prompt_text = "The key innovation of the transformer architecture is"
    prompt_tokens = tokenizer_72b.encode(prompt_text)

    # Q4 baseline: generate with all-Q4 streaming (H8a style)
    print(f"\n  Generating Q4-streaming baseline ({n_tokens} tokens)...")
    q4_index = SafetensorsBlockIndex(_find_hf_cache_path(model_id_72b))

    kv_q4 = make_prompt_cache(model_72b)
    input_ids = mx.array([prompt_tokens])
    h = inner_72b.embed_tokens(input_ids)
    mask = "causal"
    for i, layer in enumerate(inner_72b.layers):
        if i in resident_set:
            h = layer(h, mask, kv_q4[i])
        else:
            shard_cache = {}
            q4_tensors = load_block_from_safetensors(i, q4_index, shard_cache)
            assign_block_weights(blocks_72b[i], q4_tensors)
            mx.eval(blocks_72b[i].parameters())
            del shard_cache, q4_tensors
            h = layer(h, mask, kv_q4[i])
            mx.eval(h, kv_q4[i].state)
            evict_block(blocks_72b[i])
    h = inner_72b.norm(h)
    if hasattr(model_72b, "args") and model_72b.args.tie_word_embeddings:
        logits = inner_72b.embed_tokens.as_linear(h)
    else:
        logits = model_72b.lm_head(h)
    mx.eval(logits)

    # Decode Q4 baseline
    q4_nlls = []
    q4_tokens = list(prompt_tokens)
    q4_tokens.append(mx.argmax(logits[0, -1, :]).item())

    for tok_i in range(n_tokens - 1):
        next_ids = mx.array([[q4_tokens[-1]]])
        h = inner_72b.embed_tokens(next_ids)
        for i, layer in enumerate(inner_72b.layers):
            if i in resident_set:
                h = layer(h, None, kv_q4[i])
            else:
                shard_cache = {}
                q4_tensors = load_block_from_safetensors(i, q4_index, shard_cache)
                assign_block_weights(blocks_72b[i], q4_tensors)
                mx.eval(blocks_72b[i].parameters())
                del shard_cache, q4_tensors
                h = layer(h, None, kv_q4[i])
                mx.eval(h, kv_q4[i].state)
                evict_block(blocks_72b[i])
        h = inner_72b.norm(h)
        if hasattr(model_72b, "args") and model_72b.args.tie_word_embeddings:
            logits = inner_72b.embed_tokens.as_linear(h)
        else:
            logits = model_72b.lm_head(h)
        mx.eval(logits)

        log_probs = mx.log_softmax(logits[0, -1, :])
        next_token = mx.argmax(logits[0, -1, :]).item()
        nll = -log_probs[next_token].item()
        q4_nlls.append(nll)
        q4_tokens.append(next_token)

    q4_text = tokenizer_72b.decode(q4_tokens[len(prompt_tokens):])
    q4_mean_nll = float(np.mean(q4_nlls)) if q4_nlls else 0
    print(f"  Q4 baseline mean NLL: {q4_mean_nll:.4f}")
    print(f"  Q4 generated: '{q4_text[:150]}'")

    # Q2 streaming: generate with Q2 blocks
    print(f"\n  Generating Q2-streaming ({n_tokens} tokens)...")
    q2_index = Q2BlockIndex(Q2_CACHE_DIR)

    kv_q2 = make_prompt_cache(model_72b)
    input_ids = mx.array([prompt_tokens])
    h = inner_72b.embed_tokens(input_ids)
    mask = "causal"
    for i, layer in enumerate(inner_72b.layers):
        if i in resident_set:
            h = layer(h, mask, kv_q2[i])
        else:
            shard_cache = {}
            q2_tensors = load_q2_block(i, q2_index, shard_cache)
            assign_q2_block_weights(blocks_72b[i], i, q2_tensors)
            mx.eval(blocks_72b[i].parameters())
            del shard_cache, q2_tensors
            h = layer(h, mask, kv_q2[i])
            mx.eval(h, kv_q2[i].state)
            evict_block(blocks_72b[i])
            restore_q4_block_metadata(blocks_72b[i])
    h = inner_72b.norm(h)
    if hasattr(model_72b, "args") and model_72b.args.tie_word_embeddings:
        logits = inner_72b.embed_tokens.as_linear(h)
    else:
        logits = model_72b.lm_head(h)
    mx.eval(logits)

    # Decode Q2
    q2_nlls = []
    q2_tokens = list(prompt_tokens)
    q2_tokens.append(mx.argmax(logits[0, -1, :]).item())

    # Score Q2 against Q4's generated tokens (teacher forcing)
    q4_ref_tokens = q4_tokens[len(prompt_tokens):]

    for tok_i in range(min(n_tokens - 1, len(q4_ref_tokens))):
        next_ids = mx.array([[q2_tokens[-1]]])
        h = inner_72b.embed_tokens(next_ids)
        for i, layer in enumerate(inner_72b.layers):
            if i in resident_set:
                h = layer(h, None, kv_q2[i])
            else:
                shard_cache = {}
                q2_tensors = load_q2_block(i, q2_index, shard_cache)
                assign_q2_block_weights(blocks_72b[i], i, q2_tensors)
                mx.eval(blocks_72b[i].parameters())
                del shard_cache, q2_tensors
                h = layer(h, None, kv_q2[i])
                mx.eval(h, kv_q2[i].state)
                evict_block(blocks_72b[i])
                restore_q4_block_metadata(blocks_72b[i])
        h = inner_72b.norm(h)
        if hasattr(model_72b, "args") and model_72b.args.tie_word_embeddings:
            logits = inner_72b.embed_tokens.as_linear(h)
        else:
            logits = model_72b.lm_head(h)
        mx.eval(logits)

        # Score against Q4 reference token (teacher forcing)
        log_probs = mx.log_softmax(logits[0, -1, :])
        ref_token = q4_ref_tokens[tok_i]
        nll = -log_probs[ref_token].item()
        q2_nlls.append(nll)
        # Feed Q4 reference token for teacher forcing
        q2_tokens.append(ref_token)

    q2_text = tokenizer_72b.decode(q2_tokens[len(prompt_tokens):])
    q2_mean_nll = float(np.mean(q2_nlls)) if q2_nlls else 0
    nll_delta_4b = q2_mean_nll - q4_mean_nll
    print(f"  Q2 mean NLL (teacher-forced): {q2_mean_nll:.4f}")
    print(f"  NLL delta (Q2 - Q4): {nll_delta_4b:.4f}")
    print(f"  Q2 generated (teacher-forced): '{q2_text[:150]}'")

    # KL divergence on first 5 tokens (already computed — compare logit distributions)
    gate_4b = abs(nll_delta_4b) < 0.5
    print(f"\n  Gate 4b (|NLL delta| < 0.5): {'PASS' if gate_4b else 'FAIL'}")

    # Coherence check
    words = q2_text.split()
    coherent = True
    if len(words) >= 4:
        repeated = sum(1 for j in range(1, len(words)) if words[j] == words[j-1])
        coherent = repeated / len(words) < 0.5
    print(f"  Coherence check: {'PASS' if coherent else 'FAIL'}")

    overall = gate_4a and gate_4b and coherent
    print(f"\n{'='*70}")
    print(f"Phase 4 Overall: {'PASS' if overall else 'FAIL'}")
    print(f"  4a (7B PPL delta < 1.0): {'PASS' if gate_4a else 'FAIL'} ({ppl_delta_4a:.4f})")
    print(f"  4b (72B NLL delta < 0.5): {'PASS' if gate_4b else 'FAIL'} ({nll_delta_4b:.4f})")
    print(f"  Coherence: {'PASS' if coherent else 'FAIL'}")

    log_experiment(
        experiment_name="h8b_q2_streaming",
        phase="phase_4_quality_validation",
        config={
            "model_7b": model_id_7b,
            "model_72b": model_id_72b,
            "n_tokens": n_tokens,
            "calibration_text_len": len(cal_text),
        },
        results={
            "phase_4a": {
                "ppl_q4": round(ppl_q4, 4),
                "ppl_q2": round(ppl_q2, 4),
                "ppl_delta": round(ppl_delta_4a, 4),
                "nll_stability_mean": round(nll_mean, 4),
                "nll_stability_std": round(nll_std, 4),
                "gate_pass": gate_4a,
            },
            "phase_4b": {
                "q4_mean_nll": round(q4_mean_nll, 4),
                "q2_mean_nll": round(q2_mean_nll, 4),
                "nll_delta": round(nll_delta_4b, 4),
                "coherent": coherent,
                "q4_text_preview": q4_text[:200],
                "q2_text_preview": q2_text[:200],
                "gate_pass": gate_4b,
            },
            "gate_pass": overall,
        },
        env=env,
    )

    del model_72b, tokenizer_72b
    return overall


# ── Phase 5: 16 GB Provisional Projection ──


def run_phase_5(model_id: str, n_tokens: int = 20):
    """Phase 5: Simulate 16 GB memory constraint via ballast allocation."""
    print("=" * 70)
    print("Phase 5: 16 GB Provisional Projection (Stretch Goal)")
    print("=" * 70)
    print("  NOTE: Results are provisional. Final validation requires real 16 GB hardware.")

    env = get_environment_info()
    available_idle = get_available_memory_gb()
    total_ram = env["memory_gb"]
    os_overhead = total_ram - available_idle
    print(f"  Total RAM: {total_ram} GB")
    print(f"  Available (idle): {available_idle:.1f} GB")
    print(f"  OS overhead: {os_overhead:.1f} GB")

    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    # ── Config 1: 3 Q4 resident (2 first + 1 last) + 77 Q2 streamed ──
    # ── Config 2: All 80 Q2 streamed ──
    configs_16gb = [
        {"name": "3-Q4/77-Q2", "n_first": 2, "n_last": 1},
        {"name": "all-80-Q2", "n_first": 0, "n_last": 0},
    ]

    results_by_config = {}

    for cfg in configs_16gb:
        print(f"\n{'='*70}")
        print(f"Config: {cfg['name']}")
        print(f"{'='*70}")

        print(f"\n  Loading model: {model_id}")
        model, tokenizer = load(model_id)
        blocks = _get_model_blocks(model)
        inner = _get_inner_model(model)
        n_blocks = len(blocks)

        n_first = cfg["n_first"]
        n_last = cfg["n_last"]
        resident_indices = list(range(n_first)) + list(range(n_blocks - n_last, n_blocks)) if (n_first + n_last) > 0 else []
        resident_set = set(resident_indices)
        streaming_indices = [i for i in range(n_blocks) if i not in resident_set]

        print(f"  {len(resident_indices)} Q4 resident + {len(streaming_indices)} Q2 streamed")

        # Setup blocks
        for idx in range(n_blocks):
            mx.eval(blocks[idx].parameters())
            if idx not in resident_set:
                evict_block(blocks[idx])
                gc.collect()

        mx.eval(inner.embed_tokens.parameters())
        mx.eval(inner.norm.parameters())
        if hasattr(model, "lm_head"):
            mx.eval(model.lm_head.parameters())

        available_post_setup = get_available_memory_gb()
        rss_post_setup = get_rss_mb()
        metal_post_setup = mx.get_active_memory() / 1e6

        print(f"  Post-setup: RSS={rss_post_setup:.0f} MB, Avail={available_post_setup:.1f} GB, Metal={metal_post_setup:.0f} MB")

        # Check app budget constraint (10.5 GB max for 16 GB)
        app_pinned_gb = (total_ram - available_post_setup - os_overhead)
        print(f"  Estimated app pinned: {app_pinned_gb:.1f} GB")
        budget_ok = app_pinned_gb <= 10.5
        print(f"  App budget check (<= 10.5 GB): {'PASS' if budget_ok else 'FAIL'}")

        if not budget_ok:
            print(f"  WARNING: Config exceeds 10.5 GB app budget for 16 GB device")

        # Compute ballast for 16 GB simulation
        # Target: same consumed memory relative to 16 GB as measured on actual RAM
        consumed_gb = total_ram - available_post_setup
        ballast_gb = available_post_setup - (16.0 - consumed_gb)
        # Equivalently: ballast_gb = available_post_setup - 16.0 + total_ram - available_post_setup
        # = total_ram - 16.0 ... but that ignores the setup overhead.
        # Correct: we want available_after_ballast = 16.0 - consumed_gb
        target_available = 16.0 - consumed_gb

        if target_available <= 0:
            print(f"  Already consuming more than 16 GB — skipping ballast")
            print(f"  (consumed={consumed_gb:.1f} GB > 16 GB)")
            ballast_gb = 0
            ballast = (None, None)
            ballast_ok = False
        elif ballast_gb <= 0:
            print(f"  Available memory already at or below 16 GB equivalent — no ballast needed")
            ballast = (None, None)
            ballast_ok = True
        else:
            print(f"  Allocating {ballast_gb:.1f} GB ballast (target available: {target_available:.1f} GB)...")
            ballast = create_memory_pressure(target_available)
            available_after_ballast = get_available_memory_gb()
            print(f"  Available after ballast: {available_after_ballast:.1f} GB")
            # Verify ballast applied
            ballast_ok = available_after_ballast < available_post_setup - 0.5 * ballast_gb
            if not ballast_ok:
                print(f"  WARNING: Ballast verification failed — simulation may be unreliable")

        # Load Q2 index
        q2_index = Q2BlockIndex(Q2_CACHE_DIR)

        # For all-Q2 config, we need Q2 blocks for ALL layers (not just 8-71)
        # Check if Q2 index has the needed blocks
        missing_q2 = [i for i in streaming_indices if not q2_index.has_block(i)]
        if missing_q2:
            print(f"  WARNING: Q2 index missing blocks: {missing_q2}")
            print(f"  Falling back to Q4 for missing blocks")

        # Run streaming pass
        prompt = "The key innovation of the transformer architecture is"
        tokens = tokenizer.encode(prompt)
        kv_cache = make_prompt_cache(model)

        vm_before = get_vm_stat()
        avail_history = []
        rss_history = []
        per_token_ms = []
        all_load_ms = []

        # Prefill
        t_prefill = time.perf_counter()
        input_ids = mx.array([tokens])
        h = inner.embed_tokens(input_ids)
        mask = "causal"

        for i, layer in enumerate(inner.layers):
            if i in resident_set:
                h = layer(h, mask, kv_cache[i])
            elif q2_index.has_block(i):
                shard_cache = {}
                q2_tensors = load_q2_block(i, q2_index, shard_cache)
                assign_q2_block_weights(blocks[i], i, q2_tensors)
                mx.eval(blocks[i].parameters())
                del shard_cache, q2_tensors
                h = layer(h, mask, kv_cache[i])
                mx.eval(h, kv_cache[i].state)
                evict_block(blocks[i])
                restore_q4_block_metadata(blocks[i])
            else:
                # Fallback: Q4 streaming for blocks not in Q2 index
                q4_index = SafetensorsBlockIndex(_find_hf_cache_path(model_id))
                shard_cache = {}
                q4_tensors = load_block_from_safetensors(i, q4_index, shard_cache)
                assign_block_weights(blocks[i], q4_tensors)
                mx.eval(blocks[i].parameters())
                del shard_cache, q4_tensors
                h = layer(h, mask, kv_cache[i])
                mx.eval(h, kv_cache[i].state)
                evict_block(blocks[i])

        h = inner.norm(h)
        if hasattr(model, "args") and model.args.tie_word_embeddings:
            logits = inner.embed_tokens.as_linear(h)
        else:
            logits = model.lm_head(h)
        mx.eval(logits)

        next_token = mx.argmax(logits[0, -1, :]).item()
        generated = list(tokens) + [next_token]
        ttft_ms = (time.perf_counter() - t_prefill) * 1000
        print(f"\n  Prefill: {ttft_ms:.0f} ms")

        # Decode
        for tok_i in range(n_tokens - 1):
            t0 = time.perf_counter()
            input_ids = mx.array([[generated[-1]]])
            h = inner.embed_tokens(input_ids)

            token_load_ms = []
            for i, layer in enumerate(inner.layers):
                if i in resident_set:
                    h = layer(h, None, kv_cache[i])
                elif q2_index.has_block(i):
                    t_block = time.perf_counter()
                    shard_cache = {}
                    q2_tensors = load_q2_block(i, q2_index, shard_cache)
                    assign_q2_block_weights(blocks[i], i, q2_tensors)
                    mx.eval(blocks[i].parameters())
                    del shard_cache, q2_tensors
                    block_ms = (time.perf_counter() - t_block) * 1000
                    token_load_ms.append(block_ms)
                    h = layer(h, None, kv_cache[i])
                    mx.eval(h, kv_cache[i].state)
                    evict_block(blocks[i])
                    restore_q4_block_metadata(blocks[i])
                else:
                    t_block = time.perf_counter()
                    q4_index = SafetensorsBlockIndex(_find_hf_cache_path(model_id))
                    shard_cache = {}
                    q4_tensors = load_block_from_safetensors(i, q4_index, shard_cache)
                    assign_block_weights(blocks[i], q4_tensors)
                    mx.eval(blocks[i].parameters())
                    del shard_cache, q4_tensors
                    block_ms = (time.perf_counter() - t_block) * 1000
                    token_load_ms.append(block_ms)
                    h = layer(h, None, kv_cache[i])
                    mx.eval(h, kv_cache[i].state)
                    evict_block(blocks[i])

            h = inner.norm(h)
            if hasattr(model, "args") and model.args.tie_word_embeddings:
                logits = inner.embed_tokens.as_linear(h)
            else:
                logits = model.lm_head(h)
            mx.eval(logits)

            next_token = mx.argmax(logits[0, -1, :]).item()
            generated.append(next_token)

            tok_ms = (time.perf_counter() - t0) * 1000
            per_token_ms.append(tok_ms)
            all_load_ms.extend(token_load_ms)

            avail_now = get_available_memory_gb()
            rss_now = get_rss_mb()
            avail_history.append(avail_now)
            rss_history.append(rss_now)

            if tok_i < 3 or tok_i % 5 == 0 or tok_i == n_tokens - 2:
                print(f"    Token {tok_i+1}: {tok_ms:.0f} ms, RSS={rss_now:.0f} MB, Avail={avail_now:.1f} GB")

        vm_after = get_vm_stat()
        vm_d = vm_stat_delta(vm_before, vm_after)
        peak_rss = get_peak_rss_mb()
        metal_peak = mx.get_peak_memory() / 1e6

        text = tokenizer.decode(generated[len(tokens):])
        steady = per_token_ms[2:] if len(per_token_ms) > 2 else per_token_ms
        avg_ms = sum(steady) / len(steady) if steady else 0
        tok_s = 1000 / avg_ms if avg_ms > 0 else 0
        min_avail = min(avail_history) if avail_history else 0

        print(f"\n  Results for {cfg['name']}:")
        print(f"    tok/s: {tok_s:.4f}")
        print(f"    TTFT: {ttft_ms:.0f} ms")
        print(f"    Peak RSS: {peak_rss:.0f} MB")
        print(f"    Metal peak: {metal_peak:.0f} MB")
        print(f"    Min available: {min_avail:.1f} GB")
        print(f"    Pageouts: {vm_d['pageout_delta_mb']:.0f} MB")
        print(f"    Generated: '{text[:150]}'")

        # Provisional indicators (informational, not blocking)
        no_oom = True  # If we got here, no OOM
        tok_ok = tok_s >= 0.1
        avail_ok = min_avail > 1.0
        pageout_ok = vm_d["pageout_delta_mb"] < 1000

        print(f"\n    Provisional indicators:")
        print(f"      No OOM: PASS")
        print(f"      tok/s >= 0.1: {'PASS' if tok_ok else 'FAIL'} ({tok_s:.4f})")
        print(f"      min_available > 1.0 GB: {'PASS' if avail_ok else 'FAIL'} ({min_avail:.1f} GB)")
        print(f"      pageouts < 1000 MB: {'PASS' if pageout_ok else 'FAIL'} ({vm_d['pageout_delta_mb']:.0f} MB)")
        print(f"      Ballast verified: {'PASS' if ballast_ok else 'FAIL'}")

        results_by_config[cfg["name"]] = {
            "tok_s": round(tok_s, 4),
            "avg_tok_ms": round(avg_ms, 1),
            "ttft_ms": round(ttft_ms, 1),
            "peak_rss_mb": round(peak_rss, 1),
            "metal_peak_mb": round(metal_peak, 1),
            "min_available_gb": round(min_avail, 2),
            "pageout_mb": vm_d["pageout_delta_mb"],
            "pagein_mb": vm_d["pagein_delta_mb"],
            "block_load_p50_ms": round(float(np.percentile(all_load_ms, 50)), 1) if all_load_ms else 0,
            "block_load_p95_ms": round(float(np.percentile(all_load_ms, 95)), 1) if all_load_ms else 0,
            "app_pinned_gb": round(app_pinned_gb, 1),
            "budget_ok": budget_ok,
            "ballast_gb": round(ballast_gb, 1),
            "ballast_verified": ballast_ok,
            "n_resident": len(resident_indices),
            "n_streaming": len(streaming_indices),
            "n_missing_q2": len(missing_q2),
            "no_oom": no_oom,
            "tok_ok": tok_ok,
            "avail_ok": avail_ok,
            "pageout_ok": pageout_ok,
            "text_preview": text[:200],
        }

        # Release ballast
        del ballast
        del model, tokenizer
        gc.collect()

    # Summary
    print(f"\n{'='*70}")
    print(f"Phase 5 Summary — 16 GB Provisional Projection")
    print(f"{'='*70}")
    print(f"  NOTE: These are PROVISIONAL results from a {total_ram} GB machine.")
    print(f"  Final validation requires real 16 GB M4 hardware.")

    for name, r in results_by_config.items():
        indicators_passed = sum([r["no_oom"], r["tok_ok"], r["avail_ok"], r["pageout_ok"], r["ballast_verified"]])
        print(f"\n  {name}: {r['tok_s']:.4f} tok/s, {indicators_passed}/5 indicators pass")

    log_experiment(
        experiment_name="h8b_q2_streaming",
        phase="phase_5_16gb_projection",
        config={
            "model": model_id,
            "n_tokens": n_tokens,
            "total_ram_gb": total_ram,
            "os_overhead_gb": round(os_overhead, 1),
            "configs": [c["name"] for c in configs_16gb],
            "provisional": True,
        },
        results={
            "per_config": results_by_config,
            "available_idle_gb": round(available_idle, 1),
        },
        env=env,
    )


# ── Main ──


def main():
    parser = argparse.ArgumentParser(description="H8b: Q2 Streaming with Cache Optimization")
    parser.add_argument("--phase", type=str, required=True,
                        choices=["0", "0b", "1", "2", "3", "4", "5"])
    parser.add_argument("--model-7b", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    parser.add_argument("--model-72b", default="mlx-community/Qwen2.5-72B-Instruct-4bit")
    parser.add_argument("--n-tokens", type=int, default=20)
    args = parser.parse_args()

    if args.phase == "0":
        run_phase_0(args.model_72b)
    elif args.phase == "0b":
        run_phase_0b(args.model_7b)
    elif args.phase == "1":
        run_phase_1(args.model_72b)
    elif args.phase == "2":
        run_phase_2(args.model_72b, args.n_tokens)
    elif args.phase == "3":
        run_phase_3(args.model_72b, args.n_tokens)
    elif args.phase == "4":
        run_phase_4(args.model_7b, args.model_72b, args.n_tokens)
    elif args.phase == "5":
        run_phase_5(args.model_72b, args.n_tokens)


if __name__ == "__main__":
    main()
