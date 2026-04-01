"""
Phase 0: Capability Spike + Analytical Feasibility Gate for Layer LOD experiment.

0a. Toolchain validation — verify MLX mixed-quant support
0b. Tensor-level memory accounting — compute exact budgets from HuggingFace configs
0c. Model selection and baselines — download models, measure baseline metrics

Usage:
    uv run python scripts/layer_lod_phase0.py
"""

import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import psutil

# Add project root to path for experiment_utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.experiment_utils import (
    get_environment_info,
    get_rss_mb,
    log_experiment,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Qwen2.5 model configs from HuggingFace
# Source: https://huggingface.co/Qwen/Qwen2.5-{size}-Instruct/blob/main/config.json
QWEN25_CONFIGS = {
    "7B": {
        "num_hidden_layers": 28,
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "vocab_size": 152064,
        "tie_word_embeddings": False,
    },
    "14B": {
        "num_hidden_layers": 48,
        "hidden_size": 5120,
        "intermediate_size": 13824,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 152064,
        "tie_word_embeddings": False,
    },
    "72B": {
        "num_hidden_layers": 80,
        "hidden_size": 8192,
        "intermediate_size": 29568,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 152064,
        "tie_word_embeddings": False,
    },
}

# Target bit-widths for analysis
TARGET_BPWS = [2.0, 2.5, 3.0, 3.5, 4.0]

# Context length for KV cache calculation
CONTEXT_LEN = 2048
KV_BITS = 4  # Using kv_bits=4 as established in H7

# Embed/LM-head precision
EMBED_BITS = 6

# Non-model overhead initial estimate (refined after baseline load)
NON_MODEL_OVERHEAD_GB = 2.0


# ---------------------------------------------------------------------------
# Phase 0b: Tensor-level memory accounting
# ---------------------------------------------------------------------------


def compute_layer_params(cfg: dict) -> dict:
    """Compute parameter count per transformer layer for Qwen2.5 architecture.

    Each layer has:
    - Self-attention: Q, K, V projections + output projection
    - MLP: gate_proj, up_proj, down_proj (SwiGLU)
    - Layer norms (negligible)
    """
    h = cfg["hidden_size"]
    i = cfg["intermediate_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]

    # Attention
    q_params = h * (n_heads * head_dim)  # q_proj
    k_params = h * (n_kv_heads * head_dim)  # k_proj (GQA)
    v_params = h * (n_kv_heads * head_dim)  # v_proj (GQA)
    o_params = (n_heads * head_dim) * h  # o_proj
    attn_params = q_params + k_params + v_params + o_params

    # MLP (SwiGLU): gate_proj + up_proj + down_proj
    mlp_params = h * i + h * i + i * h  # gate + up + down

    # Layer norms (negligible but counted)
    norm_params = 2 * h  # input_layernorm + post_attention_layernorm

    return {
        "attention": attn_params,
        "mlp": mlp_params,
        "norm": norm_params,
        "total": attn_params + mlp_params + norm_params,
    }


def compute_memory_budget(cfg: dict, avg_trunk_bpw: float) -> dict:
    """Compute tensor-level memory budget for a model at a given avg trunk bit-width.

    Returns sizes in bytes.
    """
    layer_params = compute_layer_params(cfg)
    n_layers = cfg["num_hidden_layers"]
    vocab_size = cfg["vocab_size"]
    hidden_size = cfg["hidden_size"]

    # Trunk weights (transformer layers)
    trunk_params = layer_params["total"] * n_layers
    trunk_bytes = int(trunk_params * avg_trunk_bpw / 8)

    # Quantization metadata: ~0.5 bit/param for group scales + zero points
    # at group_size=64: 1 scale + 1 zero per group = 2 * 16bit / 64 params = 0.5 bpw overhead
    quant_meta_bytes = int(trunk_params * 0.5 / 8)

    # Embeddings (untied)
    embed_params = vocab_size * hidden_size
    embed_bytes = int(embed_params * EMBED_BITS / 8)

    # LM head (separate from embeddings for Qwen2.5)
    lm_head_params = vocab_size * hidden_size
    lm_head_bytes = int(lm_head_params * EMBED_BITS / 8)

    # KV cache at 2K context with kv_bits=4
    n_kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg["head_dim"]
    kv_bytes = int(2 * n_layers * n_kv_heads * head_dim * CONTEXT_LEN * KV_BITS / 8)

    # Total deterministic model bytes
    model_bytes = trunk_bytes + quant_meta_bytes + embed_bytes + lm_head_bytes

    return {
        "trunk_params": trunk_params,
        "trunk_bytes": trunk_bytes,
        "quant_meta_bytes": quant_meta_bytes,
        "embed_params": embed_params,
        "embed_bytes": embed_bytes,
        "lm_head_params": lm_head_params,
        "lm_head_bytes": lm_head_bytes,
        "kv_bytes": kv_bytes,
        "model_bytes": model_bytes,
        "total_bytes": model_bytes + kv_bytes,
        "params_per_layer": layer_params["total"],
        "layer_breakdown": layer_params,
    }


def run_phase_0b():
    """Phase 0b: Compute tensor-level memory accounting for all models."""
    print("\n" + "=" * 70)
    print("PHASE 0b: Tensor-Level Memory Accounting")
    print("=" * 70)

    available_gb = psutil.virtual_memory().available / (1024**3)
    print(f"\nCurrent available memory: {available_gb:.1f} GB")

    results = {}
    for model_name, cfg in QWEN25_CONFIGS.items():
        print(f"\n--- Qwen2.5-{model_name} ---")
        print(f"  Layers: {cfg['num_hidden_layers']}, Hidden: {cfg['hidden_size']}, "
              f"Intermediate: {cfg['intermediate_size']}")
        print(f"  Attention heads: {cfg['num_attention_heads']}, KV heads: {cfg['num_key_value_heads']}")
        print(f"  Vocab: {cfg['vocab_size']}, Tied embeddings: {cfg['tie_word_embeddings']}")

        model_results = {}
        for bpw in TARGET_BPWS:
            budget = compute_memory_budget(cfg, bpw)
            model_results[bpw] = budget

            total_with_overhead = budget["total_bytes"] / (1024**3) + NON_MODEL_OVERHEAD_GB
            fits = total_with_overhead < available_gb
            headroom = available_gb - total_with_overhead

            print(f"\n  At {bpw:.1f} bpw:")
            print(f"    Trunk weights:    {budget['trunk_bytes'] / (1024**3):.2f} GB "
                  f"({budget['trunk_params'] / 1e9:.2f}B params)")
            print(f"    Quant metadata:   {budget['quant_meta_bytes'] / (1024**3):.2f} GB")
            print(f"    Embeddings (Q6):  {budget['embed_bytes'] / (1024**3):.2f} GB "
                  f"({budget['embed_params'] / 1e6:.0f}M params)")
            print(f"    LM head (Q6):     {budget['lm_head_bytes'] / (1024**3):.2f} GB "
                  f"({budget['lm_head_params'] / 1e6:.0f}M params)")
            print(f"    KV cache (2K, 4b):{budget['kv_bytes'] / (1024**3):.2f} GB")
            print(f"    Non-model overhead: {NON_MODEL_OVERHEAD_GB:.1f} GB (estimated)")
            print(f"    ---")
            print(f"    Total:            {total_with_overhead:.2f} GB "
                  f"{'✓ FITS' if fits else '✗ EXCEEDS'} "
                  f"(headroom: {headroom:+.2f} GB)")

        results[model_name] = model_results

        # Log to experiments.jsonl
        log_experiment(
            experiment_name=f"layer_lod_memory_accounting_{model_name.lower()}",
            phase="memory_accounting",
            config={
                "model": f"Qwen2.5-{model_name}",
                "model_config": cfg,
                "embed_bits": EMBED_BITS,
                "kv_bits": KV_BITS,
                "context_len": CONTEXT_LEN,
                "non_model_overhead_gb": NON_MODEL_OVERHEAD_GB,
            },
            results={
                str(bpw): {
                    "trunk_bytes": model_results[bpw]["trunk_bytes"],
                    "quant_meta_bytes": model_results[bpw]["quant_meta_bytes"],
                    "embed_bytes": model_results[bpw]["embed_bytes"],
                    "lm_head_bytes": model_results[bpw]["lm_head_bytes"],
                    "kv_bytes": model_results[bpw]["kv_bytes"],
                    "total_bytes": model_results[bpw]["total_bytes"],
                    "total_with_overhead_gb": model_results[bpw]["total_bytes"] / (1024**3) + NON_MODEL_OVERHEAD_GB,
                    "fits_in_memory": (model_results[bpw]["total_bytes"] / (1024**3) + NON_MODEL_OVERHEAD_GB) < available_gb,
                    "headroom_gb": available_gb - (model_results[bpw]["total_bytes"] / (1024**3) + NON_MODEL_OVERHEAD_GB),
                }
                for bpw in TARGET_BPWS
            },
        )

    # 72B feasibility verdict
    print("\n" + "-" * 70)
    print("72B FEASIBILITY VERDICT:")
    r72 = results["72B"]
    any_fits = False
    for bpw in TARGET_BPWS:
        total_gb = r72[bpw]["total_bytes"] / (1024**3) + NON_MODEL_OVERHEAD_GB
        headroom = available_gb - total_gb
        if headroom > 1.0:
            print(f"  ✓ At {bpw:.1f} bpw: fits with {headroom:.1f} GB headroom")
            any_fits = True
        elif headroom > 0:
            print(f"  ~ At {bpw:.1f} bpw: PROVISIONAL — only {headroom:.1f} GB headroom "
                  f"(need >1 GB, revisit after Phase 3 measured overhead)")
        else:
            print(f"  ✗ At {bpw:.1f} bpw: exceeds by {-headroom:.1f} GB")

    if not any_fits:
        print("\n  → Phase 4 downgraded to ANALYSIS-ONLY with SSD streaming recommendation")
    print("-" * 70)

    return results


# ---------------------------------------------------------------------------
# Phase 0a: Toolchain validation
# ---------------------------------------------------------------------------


def run_phase_0a():
    """Phase 0a: Validate MLX mixed-quant toolchain."""
    print("\n" + "=" * 70)
    print("PHASE 0a: Toolchain Validation")
    print("=" * 70)

    results = {
        "optiq_installed": False,
        "mlx_lm_version": None,
        "mlx_version": None,
        "mixed_quant_support": False,
        "implementation_path": None,
    }

    # Check mlx and mlx-lm versions
    import mlx_lm
    mlx_version = getattr(mx, "__version__", "unknown")
    results["mlx_version"] = mlx_version
    results["mlx_lm_version"] = mlx_lm.__version__
    print(f"\n  mlx version: {mlx_version}")
    print(f"  mlx-lm version: {mlx_lm.__version__}")

    # Check optiq
    try:
        import optiq
        results["optiq_installed"] = True
        print(f"  optiq version: {optiq.__version__}")
    except ImportError:
        print("  optiq: NOT INSTALLED")

    # Check mlx-lm mixed-quant support
    print("\n  Checking mlx_lm.convert quant_predicate support...")
    from mlx_lm.convert import convert
    import inspect
    sig = inspect.signature(convert)
    has_quant_predicate = "quant_predicate" in sig.parameters
    results["mixed_quant_support"] = has_quant_predicate
    print(f"  quant_predicate parameter: {'✓ YES' if has_quant_predicate else '✗ NO'}")

    # Check built-in recipes
    try:
        from mlx_lm.convert import QUANT_RECIPES
        if isinstance(QUANT_RECIPES, dict):
            print(f"  Built-in recipes: {list(QUANT_RECIPES.keys())}")
        else:
            print(f"  Built-in recipes: {QUANT_RECIPES}")
    except ImportError:
        print("  Built-in recipes: not found (may use different API)")

    # Determine implementation path
    if results["optiq_installed"] and results["mixed_quant_support"]:
        # Check if optiq convert pipeline works (needs torch)
        try:
            from optiq.core.sensitivity import analyze_llm_sensitivity
            results["implementation_path"] = "A"
            print("\n  → Path A: OptiQ full pipeline (sensitivity + optimization + conversion)")
        except ImportError:
            results["implementation_path"] = "B"
            print("\n  → Path B: OptiQ optimizer + custom quant_predicate (torch not available)")
            print("    (Sensitivity analysis requires torch — install mlx-optiq[convert] for Path A)")
    elif results["mixed_quant_support"]:
        results["implementation_path"] = "B"
        print("\n  → Path B: MLX per-layer config via quant_predicate")
    else:
        results["implementation_path"] = "C"
        print("\n  → Path C: Manual checkpoint assembly required")

    # Log results
    log_experiment(
        experiment_name="layer_lod_toolchain_validation",
        phase="toolchain_validation",
        config={},
        results=results,
    )

    return results


# ---------------------------------------------------------------------------
# Phase 0c: Baseline measurements
# ---------------------------------------------------------------------------


def run_phase_0c():
    """Phase 0c: Download baseline model and measure metrics."""
    print("\n" + "=" * 70)
    print("PHASE 0c: Baseline Measurements")
    print("=" * 70)

    from mlx_lm import load, generate

    # Use a pre-quantized Qwen2.5-7B from mlx-community
    model_id = "mlx-community/Qwen2.5-7B-Instruct-4bit"
    print(f"\n  Loading baseline model: {model_id}")
    print(f"  (This may download the model on first run)")

    # Measure memory before load
    rss_before = get_rss_mb()
    available_before = psutil.virtual_memory().available / (1024**3)
    mx.metal.reset_peak_memory()

    # Load model
    t0 = time.time()
    model, tokenizer = load(model_id)
    mx.eval(model.parameters())
    load_time = time.time() - t0

    # Measure memory after load
    rss_after = get_rss_mb()
    available_after = psutil.virtual_memory().available / (1024**3)
    peak_metal_mb = mx.metal.get_peak_memory() / (1024**2)

    print(f"\n  Load time: {load_time:.1f}s")
    print(f"  RSS before/after: {rss_before:.0f} / {rss_after:.0f} MB (delta: {rss_after - rss_before:.0f} MB)")
    print(f"  Peak Metal memory: {peak_metal_mb:.0f} MB")
    print(f"  Available memory before/after: {available_before:.1f} / {available_after:.1f} GB")

    # Compute non-model overhead
    # Model checkpoint size (deterministic)
    model_dir = os.path.expanduser(f"~/.cache/huggingface/hub/models--{model_id.replace('/', '--')}")
    checkpoint_bytes = 0
    for root, dirs, files in os.walk(model_dir):
        for f in files:
            if f.endswith(".safetensors"):
                checkpoint_bytes += os.path.getsize(os.path.join(root, f))

    if checkpoint_bytes == 0:
        # Try to compute from model parameters
        total_params = sum(p.size for p in mx.utils.tree_flatten(model.parameters()))
        # Assume 4-bit with group_size=64 overhead
        checkpoint_bytes = int(total_params * 4.5 / 8)
        print(f"  Estimated checkpoint size: {checkpoint_bytes / (1024**3):.2f} GB (from params)")
    else:
        print(f"  Checkpoint size on disk: {checkpoint_bytes / (1024**3):.2f} GB")

    # Non-model overhead = peak process memory - deterministic model bytes
    process_memory_mb = rss_after
    non_model_overhead_mb = process_memory_mb - (checkpoint_bytes / (1024**2))
    print(f"  Non-model overhead: {non_model_overhead_mb:.0f} MB ({non_model_overhead_mb / 1024:.2f} GB)")

    # Run a quick generation to confirm model works
    print("\n  Running test generation...")
    mx.metal.reset_peak_memory()
    prompt = "Explain the concept of mixed-precision quantization in one paragraph."
    t0 = time.time()
    response = generate(model, tokenizer, prompt=prompt, max_tokens=50, verbose=False)
    gen_time = time.time() - t0
    peak_metal_gen_mb = mx.metal.get_peak_memory() / (1024**2)
    print(f"  Generation time: {gen_time:.1f}s")
    print(f"  Peak Metal during generation: {peak_metal_gen_mb:.0f} MB")
    print(f"  Response: {response[:100]}...")

    results = {
        "model_id": model_id,
        "load_time_s": round(load_time, 1),
        "rss_before_mb": round(rss_before, 0),
        "rss_after_mb": round(rss_after, 0),
        "rss_delta_mb": round(rss_after - rss_before, 0),
        "peak_metal_mb": round(peak_metal_mb, 0),
        "peak_metal_gen_mb": round(peak_metal_gen_mb, 0),
        "available_before_gb": round(available_before, 2),
        "available_after_gb": round(available_after, 2),
        "checkpoint_bytes": checkpoint_bytes,
        "non_model_overhead_mb": round(non_model_overhead_mb, 0),
        "non_model_overhead_gb": round(non_model_overhead_mb / 1024, 2),
        "generation_works": True,
    }

    log_experiment(
        experiment_name="layer_lod_baseline_qwen7b_4bit",
        phase="baseline",
        config={"model_id": model_id},
        results=results,
    )

    # Clean up
    del model, tokenizer
    mx.metal.reset_peak_memory()

    return results


# ---------------------------------------------------------------------------
# Phase 0a proof of concept: mixed-precision loading
# ---------------------------------------------------------------------------


def run_phase_0a_poc():
    """Proof of concept: load a model with mixed bit-widths via built-in recipe."""
    print("\n" + "=" * 70)
    print("PHASE 0a PoC: Mixed-Precision Loading Test")
    print("=" * 70)

    # Try loading a pre-quantized OptiQ model from HuggingFace
    # These are Qwen3.5 (not Qwen2.5) but validate the toolchain
    from mlx_lm import load, generate

    poc_model = "mlx-community/Qwen3.5-0.8B-OptiQ-4bit"
    print(f"\n  Loading OptiQ model: {poc_model}")

    try:
        model, tokenizer = load(poc_model)
        mx.eval(model.parameters())

        # Verify it has mixed precision by checking the config
        config_path = os.path.expanduser(
            f"~/.cache/huggingface/hub/models--{poc_model.replace('/', '--')}/snapshots"
        )
        # Find the config
        found_config = False
        for root, dirs, files in os.walk(config_path):
            if "config.json" in files:
                with open(os.path.join(root, "config.json")) as f:
                    cfg = json.load(f)
                if "quantization" in cfg:
                    q = cfg["quantization"]
                    # Count per-layer overrides
                    overrides = {k: v for k, v in q.items() if isinstance(v, dict)}
                    print(f"  ✓ Model loaded with mixed quantization")
                    print(f"    Default bits: {q.get('bits', '?')}, group_size: {q.get('group_size', '?')}")
                    print(f"    Per-layer overrides: {len(overrides)} layers")
                    # Show a few examples
                    for i, (name, config) in enumerate(list(overrides.items())[:3]):
                        print(f"    Example: {name} → bits={config.get('bits', '?')}")
                    found_config = True
                    break

        # Test generation
        prompt = "Hello, how are you?"
        response = generate(model, tokenizer, prompt=prompt, max_tokens=20, verbose=False)
        print(f"  ✓ Generation works: {response[:80]}...")

        del model, tokenizer

        log_experiment(
            experiment_name="layer_lod_poc_mixed_precision",
            phase="poc",
            config={"model": poc_model},
            results={
                "mixed_precision_loads": True,
                "generation_works": True,
                "has_per_layer_overrides": found_config,
            },
        )

        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        log_experiment(
            experiment_name="layer_lod_poc_mixed_precision",
            phase="poc",
            config={"model": poc_model},
            results={"mixed_precision_loads": False, "error": str(e)},
        )
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("Layer LOD Experiment — Phase 0: Capability Spike + Feasibility Gate")
    print("=" * 70)

    env = get_environment_info()
    print(f"\nEnvironment: {env['chip']}, {env['memory_gb']} GB RAM, "
          f"{env['available_gb']} GB available")

    # Phase 0a: Toolchain validation
    toolchain = run_phase_0a()

    # Phase 0a PoC: Mixed-precision loading
    poc_success = run_phase_0a_poc()

    # Phase 0b: Memory accounting
    memory = run_phase_0b()

    # Phase 0c: Baseline measurements
    baseline = run_phase_0c()

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 0 SUMMARY")
    print("=" * 70)

    gate_pass = True

    # Check 1: Mixed-quant loading
    if poc_success:
        print("  ✓ Mixed-precision loading works (OptiQ checkpoint loaded and generated)")
    else:
        print("  ✗ Mixed-precision loading FAILED")
        gate_pass = False

    # Check 2: Memory accounting complete
    print(f"  ✓ Memory accounting computed for 7B, 14B, 72B")
    print(f"  ✓ Measured non-model overhead: {baseline['non_model_overhead_gb']:.2f} GB")

    # Check 3: Implementation path
    path = toolchain["implementation_path"]
    if path in ("A", "B"):
        print(f"  ✓ Implementation path: {path}")
    else:
        print(f"  ~ Implementation path: {path} (may require more work)")

    print(f"\n  PHASE 0 GATE: {'PASS ✓' if gate_pass else 'FAIL ✗'}")

    if not gate_pass:
        print("\n  ⚠ Phase 0 gate failed. Do not proceed to Phase 1.")
        print("  Review findings in experiments.jsonl and context.md")

    return gate_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
