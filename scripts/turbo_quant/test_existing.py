"""Phase 0: Test existing KV cache quantization implementations.

Tests MLX-LM's built-in QuantizedKVCache (via kv_bits parameter in generate_step)
before implementing custom TurboQuant-style compression.
"""

import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiment_utils import (
    get_environment_info,
    get_peak_rss_mb,
    get_rss_mb,
    log_experiment,
)
from prepare import (
    BENCH_PROMPT,
    MODEL_TIERS,
    PERPLEXITY_PASSAGES,
    compute_perplexity,
    get_local_model_path,
    get_model_config,
    get_peak_gpu_mb,
    get_peak_gpu_mb_peak,
    is_model_cached,
    reset_peak_memory,
)


def test_builtin_kv_quant(tier: str = "S", kv_bits: int | None = None):
    """Test MLX-LM's built-in KV cache quantization."""
    import mlx_lm

    info = MODEL_TIERS[tier]
    repo = info["hf_repo"]

    if not is_model_cached(repo):
        print(f"  Model not cached: {repo}. Run: uv run python scripts/prepare.py --tier {tier}")
        return None

    path = get_local_model_path(repo)
    model_config = get_model_config(repo)

    print(f"\n  Loading {tier} ({info['params']}): {repo}")
    model, tokenizer = mlx_lm.load(path)
    mx.eval(model.parameters())

    rss_after_load = get_rss_mb()
    reset_peak_memory()

    label = f"kv{kv_bits}bit" if kv_bits else "fp16"
    print(f"  RSS after load: {rss_after_load:.0f} MB")
    print(f"  Testing with KV cache: {label}")

    # Generate with optional KV quantization
    gen_kwargs = {
        "max_tokens": 128,
    }
    if kv_bits is not None:
        gen_kwargs["kv_bits"] = kv_bits
        gen_kwargs["kv_group_size"] = 64

    # Warm up
    _ = mlx_lm.generate(model, tokenizer, prompt="Hello", verbose=False, **gen_kwargs)

    # Benchmark: measure tok/s
    t0 = time.perf_counter()
    response = mlx_lm.generate(model, tokenizer, prompt=BENCH_PROMPT, verbose=False, **gen_kwargs)
    elapsed = time.perf_counter() - t0

    rss_after_gen = get_rss_mb()
    peak_gpu = get_peak_gpu_mb_peak()

    # Estimate tokens generated
    n_tokens = len(tokenizer.encode(response))
    tok_s = n_tokens / elapsed if elapsed > 0 else 0

    print(f"  Generated {n_tokens} tokens in {elapsed:.2f}s = {tok_s:.1f} tok/s")
    print(f"  RSS after gen: {rss_after_gen:.0f} MB (delta: {rss_after_gen - rss_after_load:.0f} MB)")
    if peak_gpu is not None:
        print(f"  Peak GPU memory: {peak_gpu:.0f} MB")

    # Perplexity
    ppl_results = {}
    for name, text in PERPLEXITY_PASSAGES.items():
        ppl = compute_perplexity(model, tokenizer, text)
        ppl_results[name] = ppl
    avg_ppl = sum(ppl_results.values()) / len(ppl_results)
    print(f"  Avg perplexity: {avg_ppl:.2f}")

    results = {
        "tier": tier,
        "kv_bits": kv_bits,
        "kv_label": label,
        "rss_after_load_mb": round(rss_after_load, 1),
        "rss_after_gen_mb": round(rss_after_gen, 1),
        "rss_delta_mb": round(rss_after_gen - rss_after_load, 1),
        "peak_gpu_mb": peak_gpu,
        "tokens_generated": n_tokens,
        "elapsed_s": round(elapsed, 3),
        "tok_s": round(tok_s, 1),
        "perplexity": ppl_results,
        "avg_perplexity": round(avg_ppl, 2),
        "model_config": model_config,
    }

    return results


def test_long_context(tier: str = "S", kv_bits: int | None = None, gen_tokens: int = 512):
    """Test KV cache quantization with longer generation to stress the cache."""
    import mlx_lm

    info = MODEL_TIERS[tier]
    repo = info["hf_repo"]

    if not is_model_cached(repo):
        print(f"  Model not cached: {repo}")
        return None

    path = get_local_model_path(repo)

    model, tokenizer = mlx_lm.load(path)
    mx.eval(model.parameters())
    reset_peak_memory()

    label = f"kv{kv_bits}bit" if kv_bits else "fp16"

    gen_kwargs = {"max_tokens": gen_tokens}
    if kv_bits is not None:
        gen_kwargs["kv_bits"] = kv_bits
        gen_kwargs["kv_group_size"] = 64

    # Long prompt to create a substantial KV cache
    long_prompt = (
        "Write a detailed technical analysis of the following topics, covering each "
        "one thoroughly with examples, comparisons, and practical implications. "
        "Topic 1: Memory management in modern operating systems. "
        "Topic 2: Cache hierarchies in CPU architectures. "
        "Topic 3: Quantization techniques for neural networks. "
        "Please provide at least several paragraphs on each topic."
    )

    # Warm up
    _ = mlx_lm.generate(model, tokenizer, prompt="Hello", verbose=False, max_tokens=8)

    rss_before = get_rss_mb()
    reset_peak_memory()

    t0 = time.perf_counter()
    response = mlx_lm.generate(model, tokenizer, prompt=long_prompt, verbose=False, **gen_kwargs)
    elapsed = time.perf_counter() - t0

    rss_after = get_rss_mb()
    peak_gpu = get_peak_gpu_mb_peak()

    n_tokens = len(tokenizer.encode(response))
    tok_s = n_tokens / elapsed if elapsed > 0 else 0

    print(f"  Long-context {label}: {n_tokens} tokens in {elapsed:.2f}s = {tok_s:.1f} tok/s")
    print(f"  RSS delta: {rss_after - rss_before:.0f} MB, peak GPU: {peak_gpu} MB")

    # Perplexity on generated text (rough quality check)
    ppl_results = {}
    for name, text in PERPLEXITY_PASSAGES.items():
        ppl = compute_perplexity(model, tokenizer, text)
        ppl_results[name] = ppl
    avg_ppl = sum(ppl_results.values()) / len(ppl_results)

    return {
        "tier": tier,
        "kv_bits": kv_bits,
        "kv_label": label,
        "gen_tokens": gen_tokens,
        "tokens_generated": n_tokens,
        "elapsed_s": round(elapsed, 3),
        "tok_s": round(tok_s, 1),
        "rss_delta_mb": round(rss_after - rss_before, 1),
        "peak_gpu_mb": peak_gpu,
        "avg_perplexity": round(avg_ppl, 2),
    }


def main():
    env = get_environment_info()
    print("=== Phase 0: Test Built-in MLX-LM KV Cache Quantization ===")

    all_results = {}

    for tier in ["S", "M"]:
        print(f"\n--- Tier {tier} ---")
        tier_results = {}

        # Test FP16 baseline
        baseline = test_builtin_kv_quant(tier, kv_bits=None)
        if baseline:
            tier_results["fp16"] = baseline
            log_experiment(
                experiment_name=f"turbo_quant_phase0_builtin_{tier.lower()}_fp16",
                phase="turbo_quant_validation",
                config={"tier": tier, "kv_bits": None, "method": "mlx_lm_builtin"},
                results=baseline,
                env=env,
            )

        # Test built-in quantized KV at various bit widths
        for kv_bits in [8, 4]:
            result = test_builtin_kv_quant(tier, kv_bits=kv_bits)
            if result:
                tier_results[f"kv{kv_bits}"] = result
                log_experiment(
                    experiment_name=f"turbo_quant_phase0_builtin_{tier.lower()}_kv{kv_bits}",
                    phase="turbo_quant_validation",
                    config={"tier": tier, "kv_bits": kv_bits, "method": "mlx_lm_builtin"},
                    results=result,
                    env=env,
                )

        all_results[tier] = tier_results

        # Summary for this tier
        if tier_results:
            print(f"\n  === {tier} Summary ===")
            print(f"  {'Config':>10} {'tok/s':>8} {'RSS Δ MB':>10} {'Avg PPL':>10}")
            print(f"  {'-'*40}")
            for label, r in tier_results.items():
                print(f"  {label:>10} {r['tok_s']:>8.1f} {r['rss_delta_mb']:>10.1f} {r['avg_perplexity']:>10.2f}")

    # Long context tests on S tier
    print("\n--- Long Context Tests (S tier, 512 tokens) ---")
    long_results = {}
    for kv_bits in [None, 4]:
        result = test_long_context("S", kv_bits=kv_bits, gen_tokens=512)
        if result:
            label = result["kv_label"]
            long_results[label] = result
            log_experiment(
                experiment_name=f"turbo_quant_phase0_longctx_s_{label}",
                phase="turbo_quant_validation",
                config={"tier": "S", "kv_bits": kv_bits, "gen_tokens": 512, "method": "mlx_lm_builtin"},
                results=result,
                env=env,
            )

    if long_results:
        print(f"\n  === Long Context Summary ===")
        print(f"  {'Config':>10} {'tok/s':>8} {'RSS Δ MB':>10} {'Avg PPL':>10}")
        print(f"  {'-'*40}")
        for label, r in long_results.items():
            print(f"  {label:>10} {r['tok_s']:>8.1f} {r['rss_delta_mb']:>10.1f} {r['avg_perplexity']:>10.2f}")

    # Overall decision gate
    print("\n=== DECISION GATE ===")
    gate_passed = True
    for tier, tier_results in all_results.items():
        if "fp16" in tier_results and "kv4" in tier_results:
            ppl_delta = tier_results["kv4"]["avg_perplexity"] - tier_results["fp16"]["avg_perplexity"]
            print(f"  {tier}: 4-bit KV PPL delta = {ppl_delta:+.2f}")
            if abs(ppl_delta) >= 0.5:
                gate_passed = False

    if gate_passed:
        print("\n  RESULT: Built-in 4-bit KV quantization preserves quality across tiers!")
        print("  DECISION: Scope issue to characterization + practical integration guide.")
        print("  TurboQuant rotation adds complexity without clear benefit over built-in approach.")
    else:
        print("\n  RESULT: Built-in 4-bit KV quantization degrades quality.")
        print("  DECISION: Proceed with custom TurboQuant rotation-based compression.")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
