"""
run.py — Experiment executor.

Loads model, runs inference (cold/warm/quality), collects metrics, logs JSONL.
All operations are offline-only — network I/O happens exclusively in prepare.py.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Import config knobs from benchmark.py
# ---------------------------------------------------------------------------

from benchmark import (
    COMPILE_MODEL,
    CONTEXT_LENGTH,
    EXPERIMENT_NAME,
    HYPOTHESIS,
    MAX_TOKENS,
    MODEL_REPO,
    MODEL_REVISION,
    MODEL_TIER,
    REPETITIONS,
    TIME_BUDGET_SECONDS,
)
from prepare import (
    BENCH_PROMPT,
    MODEL_TIERS,
    PERPLEXITY_PASSAGES,
    check_memory_budget,
    compute_perplexity,
    compute_ssd_read_gb,
    estimate_memory_gb,
    estimate_model_size_gb,
    get_disk_io_snapshot,
    get_environment_info,
    get_local_model_path,
    get_model_config,
    get_model_revision,
    get_peak_gpu_mb_peak,
    get_peak_rss_mb,
    is_memory_safe,
    is_model_cached,
    reset_peak_memory,
)

EXPERIMENTS_FILE = Path(__file__).parent.parent / "experiments.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_model_repo() -> str:
    """Resolve the model repo from config knobs."""
    if MODEL_REPO is not None:
        return MODEL_REPO
    if MODEL_TIER not in MODEL_TIERS:
        sys.exit(f"Unknown MODEL_TIER: {MODEL_TIER}. Choose from {list(MODEL_TIERS.keys())}")
    return MODEL_TIERS[MODEL_TIER]["hf_repo"]


def _build_config(repo: str, revision: str) -> dict:
    """Build the config section for the record."""
    return {
        "model_tier": MODEL_TIER,
        "model_repo": repo,
        "model_revision": revision,
        "max_tokens": MAX_TOKENS,
        "context_length": CONTEXT_LENGTH,
        "repetitions": REPETITIONS,
        "compile_model": COMPILE_MODEL,
        "time_budget_seconds": TIME_BUDGET_SECONDS,
    }


def _build_meta() -> dict:
    """Build the meta section for the record."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": EXPERIMENT_NAME,
        "hypothesis": HYPOTHESIS,
    }


def _log_record(record: dict):
    """Append a JSON record to experiments.jsonl."""
    with open(EXPERIMENTS_FILE, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    status = record["status"]
    name = record["meta"]["experiment"]
    if status == "completed":
        warm = record["warm"]
        quality = record["quality"]
        tok_s = warm.get("tok_s")
        ppl = quality.get("perplexity_mean")
        tok_str = f"{tok_s:.1f}" if tok_s is not None else "null"
        ppl_str = f"{ppl:.2f}" if ppl is not None else "null"
        print(f"\nLogged: {name} — {tok_str} tok/s, ppl={ppl_str}")
    else:
        print(f"\nLogged: {name} — {status}: {record.get('abort_reason', '')}")
    print(f"  -> {EXPERIMENTS_FILE}")


def _abort(status: str, reason: str, env: dict, config: dict, meta: dict, **extra):
    """Log an abort record and exit."""
    record = {
        "status": status,
        "abort_reason": reason,
        "config": config,
        "env": env,
        "meta": meta,
    }
    record.update(extra)
    _log_record(record)
    sys.exit(1)


def _generate(model, tokenizer, prompt: str, max_tokens: int) -> dict:
    """Run a single generation via stream_generate. Returns timing metrics.

    Uses mlx_lm.stream_generate to get accurate per-token metrics:
    - prompt_tps from the prefill phase
    - generation_tps from the decode phase
    - finish_reason: "stop" (EOS) or "length" (truncated)
    """
    import mlx_lm

    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    response = None
    for response in mlx_lm.stream_generate(
        model, tokenizer, prompt=formatted, max_tokens=max_tokens
    ):
        pass

    if response is None:
        return {
            "ttft": 0,
            "tok_s": 0,
            "tokens_generated": 0,
            "truncated": False,
            "total_time_s": 0,
        }

    truncated = response.finish_reason == "length"
    tokens_out = response.generation_tokens
    # TTFT = prompt processing time = prompt_tokens / prompt_tps
    ttft = response.prompt_tokens / response.prompt_tps if response.prompt_tps > 0 else 0
    tok_s = response.generation_tps
    total_time = ttft + (tokens_out / tok_s if tok_s > 0 else 0)

    return {
        "ttft": round(ttft, 4),
        "tok_s": round(tok_s, 2),
        "tokens_generated": tokens_out,
        "truncated": truncated,
        "total_time_s": round(total_time, 4),
    }


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------


def run():
    env = get_environment_info()
    meta = _build_meta()
    start_time = time.monotonic()

    # -----------------------------------------------------------------------
    # Pre-flight: resolve model, check cache, estimate memory
    # -----------------------------------------------------------------------
    print(f"=== Experiment: {EXPERIMENT_NAME} ===")
    print(f"Hypothesis: {HYPOTHESIS}\n")

    repo = _resolve_model_repo()
    print(f"Model: {repo} (tier {MODEL_TIER})")

    # Resolve revision (offline-only)
    try:
        revision = get_model_revision(repo, MODEL_REVISION)
    except FileNotFoundError as e:
        config = _build_config(repo, MODEL_REVISION or "unresolved")
        _abort("aborted_preflight", str(e), env, config, meta)
        return  # unreachable, _abort calls sys.exit

    config = _build_config(repo, revision)
    print(f"Revision: {revision}")

    # Check cache
    if not is_model_cached(repo, revision):
        _abort(
            "aborted_preflight",
            f"Model not cached: {repo} @ {revision}. Run: uv run python scripts/prepare.py",
            env, config, meta,
        )
        return

    # Read model config and estimate memory
    try:
        model_config = get_model_config(repo, revision)
    except FileNotFoundError as e:
        _abort("aborted_preflight", f"Cannot read model config: {e}", env, config, meta)
        return

    # Use tier's approx_size_gb only when repo matches the tier default.
    # When MODEL_REPO overrides, measure actual cached weight files to avoid
    # underestimating memory for a larger model.
    tier_info = MODEL_TIERS.get(MODEL_TIER, {})
    if MODEL_REPO is None and tier_info:
        approx_size = tier_info["approx_size_gb"]
    else:
        approx_size = estimate_model_size_gb(repo, revision)
    estimate = estimate_memory_gb(approx_size, model_config, CONTEXT_LENGTH)
    print(f"Memory estimate: {estimate:.1f}GB")

    ok, msg = check_memory_budget(estimate)
    if not ok:
        _abort("aborted_preflight", msg, env, config, meta)
        return

    print(f"Memory budget: {msg}\n")

    # -----------------------------------------------------------------------
    # Phase 1 — Cold (1x): load + first generation
    # -----------------------------------------------------------------------
    print("--- Phase 1: Cold run ---")
    reset_peak_memory()
    disk_before = get_disk_io_snapshot()

    import mlx.core as mx
    import mlx_lm

    # Resolve to local filesystem path to guarantee no network I/O.
    # mlx_lm.load() with a repo ID would route through snapshot_download().
    local_path = get_local_model_path(repo, revision)
    print(f"Loading from: {local_path}")

    t_load_start = time.perf_counter()
    model, tokenizer = mlx_lm.load(local_path)

    if COMPILE_MODEL:
        model = mx.compile(model)

    mx.eval(mx.zeros(1))  # force sync
    load_time = time.perf_counter() - t_load_start
    print(f"Model loaded in {load_time:.2f}s")

    # Memory check after load
    if not is_memory_safe():
        disk_after = get_disk_io_snapshot()
        peak = {"rss_mb": round(get_peak_rss_mb(), 1), "gpu_mb": get_peak_gpu_mb_peak()}
        cold_partial = {
            "load_time_s": round(load_time, 4),
            "ssd_read_gb": compute_ssd_read_gb(disk_before, disk_after),
        }
        _abort(
            "aborted_memory",
            "Memory unsafe after model load",
            env, config, meta,
            cold=cold_partial, peak=peak,
        )
        return

    # Cold generation
    cold_gen = _generate(model, tokenizer, BENCH_PROMPT, MAX_TOKENS)
    disk_after = get_disk_io_snapshot()

    cold = {
        "load_time_s": round(load_time, 4),
        "ttft": cold_gen["ttft"],
        "tok_s": cold_gen["tok_s"],
        "tokens_generated": cold_gen["tokens_generated"],
        "truncated": cold_gen["truncated"],
        "ssd_read_gb": compute_ssd_read_gb(disk_before, disk_after),
    }
    print(f"Cold: {cold['tok_s']:.1f} tok/s, {cold['tokens_generated']} tokens\n")

    # -----------------------------------------------------------------------
    # Phase 2 — Warm (N reps)
    # -----------------------------------------------------------------------
    print(f"--- Phase 2: Warm ({REPETITIONS} reps) ---")
    warm_runs = []
    abort_status = None
    abort_reason = ""

    for i in range(REPETITIONS):
        # Time budget check
        if time.monotonic() - start_time > TIME_BUDGET_SECONDS:
            abort_status = "aborted_timeout"
            abort_reason = f"Time budget exceeded ({TIME_BUDGET_SECONDS}s)"
            print(f"  Time budget exceeded after rep {i}")
            break

        run_result = _generate(model, tokenizer, BENCH_PROMPT, MAX_TOKENS)
        warm_runs.append(run_result)
        print(f"  Rep {i+1}: {run_result['tok_s']:.1f} tok/s, {run_result['tokens_generated']} tokens")

        # Memory check after each rep
        if not is_memory_safe():
            abort_status = "aborted_memory"
            abort_reason = f"Memory unsafe after warm rep {i+1}"
            print(f"  Memory unsafe after rep {i+1}")
            break

    peak = {"rss_mb": round(get_peak_rss_mb(), 1), "gpu_mb": get_peak_gpu_mb_peak()}

    if abort_status:
        # Non-truncated warm runs for stats
        valid_runs = [r for r in warm_runs if not r["truncated"]]
        warm = {
            "repetitions": len(warm_runs),
            "runs": warm_runs,
            "tok_s": round(sum(r["tok_s"] for r in valid_runs) / len(valid_runs), 2) if valid_runs else None,
            "ttft": round(sum(r["ttft"] for r in valid_runs) / len(valid_runs), 4) if valid_runs else None,
        }
        _abort(
            abort_status, abort_reason,
            env, config, meta,
            cold=cold, warm=warm, peak=peak,
        )
        return

    # Compute warm aggregates
    valid_runs = [r for r in warm_runs if not r["truncated"]]
    all_truncated = len(valid_runs) == 0

    warm = {
        "repetitions": len(warm_runs),
        "runs": warm_runs,
        "tok_s": round(sum(r["tok_s"] for r in valid_runs) / len(valid_runs), 2) if valid_runs else None,
        "ttft": round(sum(r["ttft"] for r in valid_runs) / len(valid_runs), 4) if valid_runs else None,
    }

    if all_truncated:
        print("  WARNING: All warm runs truncated — tok/s and ttft are null")
    else:
        print(f"  Warm avg: {warm['tok_s']:.1f} tok/s, ttft={warm['ttft']:.4f}s\n")

    # -----------------------------------------------------------------------
    # Phase 3 — Quality (perplexity)
    # -----------------------------------------------------------------------
    print("--- Phase 3: Quality (perplexity) ---")

    # Memory check before starting quality
    if not is_memory_safe():
        _abort(
            "aborted_memory", "Memory unsafe before quality eval",
            env, config, meta,
            cold=cold, warm=warm, peak=peak,
        )
        return

    perplexity_passages = {}
    quality_abort_status = None
    quality_abort_reason = ""

    for name, text in PERPLEXITY_PASSAGES.items():
        # Time budget check
        if time.monotonic() - start_time > TIME_BUDGET_SECONDS:
            quality_abort_status = "aborted_timeout"
            quality_abort_reason = f"Time budget exceeded during quality ({name})"
            break

        ppl = compute_perplexity(model, tokenizer, text)
        perplexity_passages[name] = ppl
        print(f"  {name}: ppl={ppl:.2f}")

        # Memory check between passages
        if not is_memory_safe():
            quality_abort_status = "aborted_memory"
            quality_abort_reason = f"Memory unsafe after quality passage '{name}'"
            break

    peak = {"rss_mb": round(get_peak_rss_mb(), 1), "gpu_mb": get_peak_gpu_mb_peak()}

    if quality_abort_status:
        quality_partial = {
            "perplexity_passages": perplexity_passages,
            "perplexity_mean": (
                round(sum(perplexity_passages.values()) / len(perplexity_passages), 4)
                if perplexity_passages else None
            ),
            "cache_hit_rate": None,
        }
        _abort(
            quality_abort_status, quality_abort_reason,
            env, config, meta,
            cold=cold, warm=warm, peak=peak, quality=quality_partial,
        )
        return

    # Compute quality aggregates
    ppl_mean = round(
        sum(perplexity_passages.values()) / len(perplexity_passages), 4
    ) if perplexity_passages else None

    quality = {
        "perplexity_mean": ppl_mean,
        "perplexity_passages": perplexity_passages,
        "cache_hit_rate": None,
    }
    print(f"  Mean perplexity: {ppl_mean:.2f}\n")

    # -----------------------------------------------------------------------
    # Completed record
    # -----------------------------------------------------------------------
    record = {
        "status": "completed",
        "cold": cold,
        "warm": warm,
        "peak": peak,
        "quality": quality,
        "config": config,
        "env": env,
        "meta": meta,
    }
    _log_record(record)


if __name__ == "__main__":
    run()
