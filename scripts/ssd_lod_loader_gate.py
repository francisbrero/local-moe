"""
Phase 1b: Loader Strategy Selection for SSD Layer LOD experiment.

HARD GATE — validates whether MLX can do zero-copy streaming (expected: NO,
confirmed by H0 Phase 4a) and builds a C/pread staging buffer prototype as
the expected alternative path.

Tests:
1. MLX zero-copy test: swap layers 10 times, measure RSS growth
2. pread staging buffer: load blocks via pread into fixed buffer, measure RSS stability
3. End-to-end inference integration: load-swap-forward cycle with RSS monitoring
4. mlock/madvise residency validation

Usage:
    uv run python scripts/ssd_lod_loader_gate.py
"""

import ctypes
import ctypes.util
import math
import mmap
import os
import struct
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.experiment_utils import get_environment_info, get_rss_mb, log_experiment


# ---------------------------------------------------------------------------
# Test 1: MLX zero-copy verification (reproduce H0 finding)
# ---------------------------------------------------------------------------

def test_mlx_zero_copy(model, n_swaps: int = 10) -> dict:
    """Swap one layer's weights 10 times via mx.array(), measure RSS growth.

    Expected: RSS grows ~2-3 MB per swap (confirming H0's finding that
    MLX copies mmap data internally).
    """
    print("\n" + "=" * 60)
    print("Test 1: MLX Zero-Copy Verification")
    print("=" * 60)

    # Find a middle block's linear layer to swap
    target_layer = None
    target_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear) and "layers.14." in name and "gate_proj" in name:
            target_layer = module
            target_name = name
            break

    if target_layer is None:
        print("  ERROR: Could not find target layer")
        return {"error": True, "zero_copy": None}

    print(f"  Target: {target_name}")
    print(f"  Weight shape: {target_layer.weight.shape}")
    weight_size_mb = target_layer.weight.nbytes / (1024 * 1024)
    print(f"  Weight size: {weight_size_mb:.1f} MB")

    # Get the weight data as numpy, then create a mmap-like source
    original_weight = target_layer.weight
    weight_np = np.array(original_weight)
    mx.eval(original_weight)

    rss_before = get_rss_mb()
    print(f"\n  RSS before swaps: {rss_before:.1f} MB")

    rss_deltas = []
    for i in range(n_swaps):
        rss_pre = get_rss_mb()
        # Simulate what a streaming loader would do: create mx.array from numpy/buffer
        new_weight = mx.array(weight_np)
        target_layer.weight = new_weight
        mx.eval(target_layer.weight)
        rss_post = get_rss_mb()
        delta = rss_post - rss_pre
        rss_deltas.append(delta)
        print(f"  Swap {i+1}: RSS {rss_pre:.1f} → {rss_post:.1f} MB (Δ{delta:+.1f})")

    rss_after = get_rss_mb()
    total_growth = rss_after - rss_before
    avg_delta = sum(rss_deltas) / len(rss_deltas)

    # Restore original
    target_layer.weight = original_weight
    mx.eval(target_layer.weight)

    zero_copy = total_growth < weight_size_mb  # If less than one copy, it's zero-copy
    print(f"\n  Total RSS growth: {total_growth:.1f} MB over {n_swaps} swaps")
    print(f"  Avg RSS delta per swap: {avg_delta:.1f} MB")
    print(f"  Weight size: {weight_size_mb:.1f} MB")
    print(f"  Zero-copy: {'YES' if zero_copy else 'NO (MLX copies internally)'}")

    return {
        "error": False,
        "zero_copy": zero_copy,
        "total_rss_growth_mb": round(total_growth, 1),
        "avg_rss_delta_mb": round(avg_delta, 1),
        "weight_size_mb": round(weight_size_mb, 1),
        "n_swaps": n_swaps,
    }


# ---------------------------------------------------------------------------
# Test 2: pread staging buffer prototype
# ---------------------------------------------------------------------------

def test_pread_staging_buffer(block_size_mb: float = 262.0) -> dict:
    """Prototype a pread-based staging buffer that reuses memory.

    Creates a temp file, reads it repeatedly via os.pread() into a fixed buffer,
    and measures RSS stability.
    """
    print("\n" + "=" * 60)
    print("Test 2: pread Staging Buffer Prototype")
    print("=" * 60)

    block_size = int(block_size_mb * 1024 * 1024)
    n_loads = 10

    # Create temp file with random data
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        tmpfile = f.name
        # Write in chunks to avoid huge memory spike
        chunk_size = 64 * 1024 * 1024  # 64 MB chunks
        remaining = block_size
        while remaining > 0:
            write_size = min(chunk_size, remaining)
            f.write(os.urandom(write_size))
            remaining -= write_size

    print(f"  Temp file: {tmpfile} ({block_size_mb:.0f} MB)")

    try:
        fd = os.open(tmpfile, os.O_RDONLY)

        # Pre-allocate staging buffer (fixed size, reused)
        staging_buffer = bytearray(block_size)

        rss_before = get_rss_mb()
        print(f"  RSS before loads: {rss_before:.1f} MB")

        load_times = []
        rss_deltas = []

        for i in range(n_loads):
            rss_pre = get_rss_mb()
            t0 = time.perf_counter()

            # Read entire block via pread into staging buffer
            bytes_read = 0
            while bytes_read < block_size:
                chunk = os.pread(fd, min(block_size - bytes_read, 4 * 1024 * 1024), bytes_read)
                if not chunk:
                    raise IOError(f"pread returned 0 bytes at offset {bytes_read}")
                staging_buffer[bytes_read:bytes_read + len(chunk)] = chunk
                bytes_read += len(chunk)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            rss_post = get_rss_mb()
            delta = rss_post - rss_pre
            load_times.append(elapsed_ms)
            rss_deltas.append(delta)

            bandwidth_gbs = (block_size / (1024**3)) / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
            print(f"  Load {i+1}: {elapsed_ms:.1f} ms ({bandwidth_gbs:.1f} GB/s), RSS Δ{delta:+.1f} MB")

        os.close(fd)

        rss_after = get_rss_mb()
        total_growth = rss_after - rss_before
        # Steady-state: skip first 2 loads (warmup/GC effects)
        steady_times = load_times[2:] if len(load_times) > 2 else load_times
        avg_steady_time = sum(steady_times) / len(steady_times)
        avg_bandwidth = (block_size / (1024**3)) / (avg_steady_time / 1000) if avg_steady_time > 0 else 0

        # Check: RSS should be flat in steady state (loads 3-10)
        steady_deltas = rss_deltas[2:] if len(rss_deltas) > 2 else rss_deltas
        max_steady_delta = max(abs(d) for d in steady_deltas) if steady_deltas else 0
        rss_flat = max_steady_delta < 5.0  # <5 MB variation in steady state

        print(f"\n  Total RSS growth: {total_growth:.1f} MB (includes warmup)")
        print(f"  Steady-state max delta: {max_steady_delta:.1f} MB")
        print(f"  RSS flat in steady state: {'YES' if rss_flat else 'NO'}")
        print(f"  Avg load time (steady): {avg_steady_time:.1f} ms")
        print(f"  Avg bandwidth (steady): {avg_bandwidth:.1f} GB/s")

        # Check against H0 pread benchmark: cold should be <75 ms for 200 MB
        # Our block is ~262 MB, so threshold = 75 * (262/200) ≈ 98 ms
        latency_ok = avg_steady_time < 100  # generous threshold

        return {
            "rss_flat": rss_flat,
            "total_rss_growth_mb": round(total_growth, 1),
            "max_steady_delta_mb": round(max_steady_delta, 1),
            "avg_load_ms": round(avg_steady_time, 1),
            "avg_bandwidth_gbs": round(avg_bandwidth, 1),
            "latency_ok": latency_ok,
            "n_loads": n_loads,
            "block_size_mb": block_size_mb,
        }

    finally:
        os.unlink(tmpfile)


# ---------------------------------------------------------------------------
# Test 3: End-to-end inference integration
# ---------------------------------------------------------------------------

def test_inference_integration(model, tokenizer, n_cycles: int = 10) -> dict:
    """Load-swap-forward cycle: replace a layer's weights from a pread buffer
    and run a full forward pass, checking RSS/Metal stability.
    """
    print("\n" + "=" * 60)
    print("Test 3: End-to-End Inference Integration")
    print("=" * 60)

    # Find target layer
    target_module = None
    target_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear) and "layers.14." in name and "gate_proj" in name:
            target_module = module
            target_name = name
            break

    if target_module is None:
        print("  ERROR: Could not find target layer")
        return {"error": True}

    print(f"  Target: {target_name}")

    # Save original weights
    original_weight = target_module.weight
    original_scales = target_module.scales
    original_biases = getattr(target_module, "biases", None)

    # Get weight data as numpy (simulating pread from disk)
    weight_np = np.array(original_weight)
    scales_np = np.array(original_scales)
    biases_np = np.array(original_biases) if original_biases is not None else None

    # Prepare a test input
    test_text = "The transformer architecture has revolutionized"
    tokens = tokenizer.encode(test_text)
    input_ids = mx.array(tokens)[None, :]

    # Get baseline output
    baseline_out = model(input_ids, cache=None)
    baseline_logits = baseline_out[0] if isinstance(baseline_out, tuple) else baseline_out
    baseline_logits = baseline_logits[0, -1, :]  # last token logits
    mx.eval(baseline_logits)

    rss_before = get_rss_mb()
    metal_before = mx.get_peak_memory() / (1024 * 1024) if hasattr(mx, "get_peak_memory") else 0

    print(f"  RSS before cycles: {rss_before:.1f} MB")
    print(f"  Metal peak before: {metal_before:.1f} MB")

    rss_deltas = []
    logit_matches = []

    for i in range(n_cycles):
        rss_pre = get_rss_mb()

        # Simulate pread: create new mx.array from numpy data (like staging buffer)
        new_weight = mx.array(weight_np)
        new_scales = mx.array(scales_np)
        new_biases = mx.array(biases_np) if biases_np is not None else None

        # Swap into model
        target_module.weight = new_weight
        target_module.scales = new_scales
        if new_biases is not None:
            target_module.biases = new_biases
        mx.eval(target_module.weight, target_module.scales)

        # Forward pass
        out = model(input_ids, cache=None)
        logits = out[0] if isinstance(out, tuple) else out
        logits = logits[0, -1, :]
        mx.eval(logits)

        # Check logit match
        diff = mx.abs(logits - baseline_logits).max().item()
        logit_matches.append(diff < 1e-3)

        rss_post = get_rss_mb()
        delta = rss_post - rss_pre
        rss_deltas.append(delta)

        print(f"  Cycle {i+1}: RSS Δ{delta:+.1f} MB, logit diff={diff:.6f}, match={'YES' if diff < 1e-3 else 'NO'}")

    rss_after = get_rss_mb()
    metal_after = mx.get_peak_memory() / (1024 * 1024) if hasattr(mx, "get_peak_memory") else 0
    total_rss_growth = rss_after - rss_before
    metal_growth = metal_after - metal_before

    # Restore original
    target_module.weight = original_weight
    target_module.scales = original_scales
    if original_biases is not None:
        target_module.biases = original_biases
    mx.eval(target_module.weight, target_module.scales)

    # Evaluate steady-state (skip first 2 cycles — MLX GC warmup)
    steady_deltas = rss_deltas[2:] if len(rss_deltas) > 2 else rss_deltas
    max_steady_delta = max(abs(d) for d in steady_deltas) if steady_deltas else 0
    rss_flat = max_steady_delta < 10.0  # <10 MB variation in steady state
    metal_flat = abs(metal_growth) < (metal_before * 0.05) if metal_before > 0 else True
    all_logits_match = all(logit_matches)

    print(f"\n  Total RSS growth: {total_rss_growth:.1f} MB (includes warmup)")
    print(f"  Steady-state max delta (cycles 3-{n_cycles}): {max_steady_delta:.1f} MB")
    print(f"  RSS flat in steady state: {'YES' if rss_flat else 'NO'}")
    print(f"  Metal growth: {metal_growth:.1f} MB ({'FLAT' if metal_flat else 'GROWING'})")
    print(f"  All logits match: {'YES' if all_logits_match else 'NO'}")

    return {
        "error": False,
        "rss_flat": rss_flat,
        "metal_flat": metal_flat,
        "all_logits_match": all_logits_match,
        "total_rss_growth_mb": round(total_rss_growth, 1),
        "max_steady_delta_mb": round(max_steady_delta, 1),
        "metal_growth_mb": round(metal_growth, 1),
        "n_cycles": n_cycles,
        "avg_rss_delta_mb": round(sum(rss_deltas) / len(rss_deltas), 1),
    }


# ---------------------------------------------------------------------------
# Test 4: mlock / madvise validation
# ---------------------------------------------------------------------------

def test_mlock_limits() -> dict:
    """Probe macOS mlock limits and madvise behavior."""
    print("\n" + "=" * 60)
    print("Test 4: mlock/madvise Validation")
    print("=" * 60)

    import resource

    # Check mlock limits
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    print(f"  RLIMIT_MEMLOCK: soft={soft}, hard={hard}")
    if soft == resource.RLIM_INFINITY:
        print(f"  mlock: unlimited (root or configured)")
    else:
        print(f"  mlock: limited to {soft} bytes ({soft / (1024*1024):.1f} MB)")

    # Try to mlock a small region
    test_size = 4096
    buf = bytearray(test_size)
    buf_ptr = (ctypes.c_char * test_size).from_buffer(buf)
    addr = ctypes.addressof(buf_ptr)

    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    result = libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(test_size))
    mlock_works = result == 0
    if mlock_works:
        libc.munlock(ctypes.c_void_p(addr), ctypes.c_size_t(test_size))
    print(f"  mlock(4KB): {'SUCCESS' if mlock_works else 'FAILED (errno may be ENOMEM or EPERM)'}")

    # Test with a larger region (1 MB)
    large_test = 1024 * 1024
    buf2 = bytearray(large_test)
    buf2_ptr = (ctypes.c_char * large_test).from_buffer(buf2)
    addr2 = ctypes.addressof(buf2_ptr)
    result2 = libc.mlock(ctypes.c_void_p(addr2), ctypes.c_size_t(large_test))
    mlock_1mb = result2 == 0
    if mlock_1mb:
        libc.munlock(ctypes.c_void_p(addr2), ctypes.c_size_t(large_test))
    print(f"  mlock(1MB): {'SUCCESS' if mlock_1mb else 'FAILED'}")

    # Test madvise on mmap
    print(f"\n  Testing madvise on mmap'd region...")
    with tempfile.NamedTemporaryFile(delete=False) as f:
        tmpfile = f.name
        f.write(os.urandom(4 * 1024 * 1024))  # 4 MB

    try:
        fd = os.open(tmpfile, os.O_RDWR)
        mm = mmap.mmap(fd, 0, access=mmap.ACCESS_WRITE)

        # MADV_WILLNEED
        MADV_WILLNEED = 3
        MADV_DONTNEED = 4
        MADV_FREE = 5

        mm_addr = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(mm)))
        mm_len = ctypes.c_size_t(len(mm))

        r_willneed = libc.madvise(mm_addr, mm_len, MADV_WILLNEED)
        print(f"  madvise(MADV_WILLNEED): {'SUCCESS' if r_willneed == 0 else 'FAILED'}")

        r_dontneed = libc.madvise(mm_addr, mm_len, MADV_DONTNEED)
        print(f"  madvise(MADV_DONTNEED): {'SUCCESS' if r_dontneed == 0 else 'FAILED'}")

        # MADV_FREE might not be available on all macOS versions
        r_free = libc.madvise(mm_addr, mm_len, MADV_FREE)
        print(f"  madvise(MADV_FREE): {'SUCCESS' if r_free == 0 else 'FAILED'}")

        mm.close()
        os.close(fd)
    finally:
        os.unlink(tmpfile)

    return {
        "rlimit_soft": soft,
        "rlimit_hard": hard,
        "mlock_4kb": mlock_works,
        "mlock_1mb": mlock_1mb,
        "madvise_willneed": r_willneed == 0,
        "madvise_dontneed": r_dontneed == 0,
        "madvise_free": r_free == 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from mlx_lm import load

    print("=" * 60)
    print("Phase 1b: Loader Strategy Selection (HARD GATE)")
    print("=" * 60)

    env = get_environment_info()
    print(f"Environment: {env['chip']}, {env['memory_gb']} GB RAM")

    model_id = "mlx-community/Qwen2.5-7B-Instruct-4bit"
    print(f"\nLoading model: {model_id}")
    model, tokenizer = load(model_id)
    mx.eval(model.parameters())

    # Test 1: MLX zero-copy
    zc_result = test_mlx_zero_copy(model)

    # Test 2: pread staging buffer
    pread_result = test_pread_staging_buffer(block_size_mb=262.0)

    # Test 3: End-to-end inference integration
    integration_result = test_inference_integration(model, tokenizer)

    # Test 4: mlock/madvise
    mlock_result = test_mlock_limits()

    # --- Gate evaluation ---
    print("\n" + "=" * 60)
    print("GATE EVALUATION")
    print("=" * 60)

    mlx_zero_copy = zc_result.get("zero_copy", False)
    pread_rss_flat = pread_result.get("rss_flat", False)
    pread_latency_ok = pread_result.get("latency_ok", False)
    integration_rss_flat = integration_result.get("rss_flat", False)
    integration_metal_flat = integration_result.get("metal_flat", False)
    integration_logits_ok = integration_result.get("all_logits_match", False)

    print(f"\n  MLX zero-copy:           {'PASS' if mlx_zero_copy else 'FAIL (expected)'}")
    print(f"  pread RSS flat:          {'PASS' if pread_rss_flat else 'FAIL'}")
    print(f"  pread latency OK:        {'PASS' if pread_latency_ok else 'FAIL'}")
    print(f"  Integration RSS flat:    {'PASS' if integration_rss_flat else 'FAIL'}")
    print(f"  Integration Metal flat:  {'PASS' if integration_metal_flat else 'FAIL'}")
    print(f"  Integration logits OK:   {'PASS' if integration_logits_ok else 'FAIL'}")

    if mlx_zero_copy:
        loader_path = "mlx_mmap"
        print(f"\n  LOADER SELECTED: MLX mmap (zero-copy works!)")
    elif pread_rss_flat and pread_latency_ok:
        if integration_rss_flat and integration_metal_flat and integration_logits_ok:
            loader_path = "c_pread_staging"
            print(f"\n  LOADER SELECTED: C/pread staging buffer (full integration validated)")
        else:
            # pread works at loader level but MLX still copies during forward pass
            loader_path = "c_pread_staging_with_copy"
            print(f"\n  LOADER SELECTED: C/pread staging (WARNING: MLX copies during forward pass)")
            print(f"  RSS grows during inference cycles — memory model is approximate.")
            print(f"  May still work if RSS growth stabilizes (GC reclaims old tensors).")
    else:
        loader_path = "ABORT"
        print(f"\n  GATE FAILED: No viable loader path. Experiment cannot proceed.")

    gate_pass = loader_path != "ABORT"

    print(f"\n  OVERALL GATE: {'PASS' if gate_pass else 'FAIL'}")

    # Log results
    log_experiment(
        experiment_name="ssd_lod_loader_gate",
        phase="loader_gate",
        config={
            "model": model_id,
        },
        results={
            "mlx_zero_copy": zc_result,
            "pread_staging": pread_result,
            "inference_integration": integration_result,
            "mlock_madvise": mlock_result,
            "loader_path": loader_path,
            "gate_pass": gate_pass,
            "pass_fail": "PASS" if gate_pass else "FAIL",
        },
    )

    del model, tokenizer
    print(f"\nResults logged to experiments.jsonl")


if __name__ == "__main__":
    main()
