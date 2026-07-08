"""
h9_e2_kv_workload.py — KV quantization at workload context lengths (issue #36, H9-E2 Arm a).

Validates H7's kv4 result (`kv_bits=4, kv_group_size=64`, zero PPL loss) on the REAL Tier-1
model (Qwen3-30B-A3B-4bit) at the REAL workload context lengths (10K/12K/15K-token
call-transcript summarization), and quantifies effective-context-per-GB of KV cache.

Scope (plan.md): in-process mlx_lm engine tier, idle 24 GB M4 Pro (off-hours tier). NOT the
16 GB daily-driver budget; NOT the vllm-mlx served tier (follow-up). Arm (b) llama.cpp is
deferred (blocked on E1b host contention).

Method (see dev/active/issue-36-h9-e2-kv-quant/plan.md):
  - Synthetic multi-speaker meeting transcripts, deterministically seeded by sample index,
    tokenizer-trimmed to exactly 10240 / 12288 / 15360 tokens.
  - Per (length, config in {kv16, kv4}):
      * Two-stage memory preflight (pre-load total-budget + post-load incremental headroom),
        no double-counting of weights. Skip + log if it won't fit.
      * Prefill the transcript into a prompt_cache; apply maybe_quantize_kv_cache for kv4
        (quantized_kv_start=0); ASSERT every entry is QuantizedKVCache for kv4.
      * KV-cache size read DIRECTLY via sum(c.nbytes) (gate metric), cross-checked vs the
        analytic formula (hard validity gate: must agree within 25%).
      * Quality: pin the kv16 greedy summary as the target, teacher-force it through the
        SAME cached path under both configs, score NLL on summary-token positions only
        (summary-only ΔPPL is the formal gate). ROUGE-L is a divergence trigger, not a gate.
      * Speed: prefill tok/s, decode tok/s, TTFT (reporting; >10% kv4 slowdown flagged).
  - Gates: summary-only |ΔPPL| <= 1% rel + spot-check; effective ctx/GB >= 1.5x; no OOM @15K.

Usage:
  uv run python scripts/h9_e2_kv_workload.py smoke           # Phase A de-risk + compat gate
  uv run python scripts/h9_e2_kv_workload.py run             # full sweep (all lengths)
  uv run python scripts/h9_e2_kv_workload.py run --lengths 10240 --samples 2   # quick subset
"""

from __future__ import annotations

import argparse
import gc
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import psutil

sys.path.insert(0, str(Path(__file__).parent))

from h9_e1_baseline import tree_footprint_gb  # noqa: E402
from experiment_utils import (  # noqa: E402
    get_available_memory_gb,
    get_environment_info,
    log_experiment,
)

MODEL = "mlx-community/Qwen3-30B-A3B-4bit"
LOGDIR = Path(__file__).parent.parent / "dev" / "active" / "issue-36-h9-e2-kv-quant" / "logs"
SUMMARY_DIR = LOGDIR / "summaries"
TRANSCRIPT_DIR = LOGDIR / "transcripts"

# Pinned configuration (plan.md "Pinned configuration") — every value logged.
LENGTHS = [10240, 12288, 15360]
KV_GROUP_SIZE = 64
QUANTIZED_KV_START = 0
MAX_TOKENS = 256
SAMPLES_PER_LENGTH = 5
WEIGHTS_GB = 16.55          # E1-measured MLX-wired footprint of this model
SCRATCH_GB = 2.0           # MLX/Metal prefill scratch margin at 15K
SAFETY_GB = 1.0
USABLE_CAP_GB = 20.0       # idle 24 GB host usable cap (conservative within 19-21 GB)
NBYTES_ANALYTIC_TOL = 0.25 # hard validity gate: .nbytes vs analytic must agree within 25%
PPL_GATE_REL = 0.01        # |ΔPPL| <= 1% relative
PERGB_GATE = 1.5           # effective ctx/GB >= 1.5x
ROUGE_REVIEW_FLOOR = 0.90  # below this, escalate spot-check (not a hard fail)
SPEED_FLAG_REL = 0.10      # flag if kv4 >10% slower than kv16

SUMMARIZE_INSTRUCTION = (
    "You are an operations assistant. Read the following meeting transcript and write a "
    "concise, well-structured summary covering the key decisions, owners, dates, open "
    "questions, and next actions. Be specific and faithful to the transcript.\n\n"
    "TRANSCRIPT:\n"
)


# ---------------------------------------------------------------------------
# Synthetic multi-speaker meeting transcripts (deterministic, seeded by index)
# ---------------------------------------------------------------------------

_SPEAKERS = ["Alex", "Priya", "Diego", "Sam", "Mei", "Jordan"]
_TOPICS = [
    ("quarterly planning", [
        "we need to lock the Q3 roadmap before the board sync on the 14th",
        "the infra migration is the long pole; it slips two weeks if hiring stalls",
        "marketing wants the launch gated on the analytics dashboard being live",
        "let's commit to three OKRs, not seven, and actually fund them",
        "the budget delta from last quarter is about 12 percent, mostly cloud",
    ]),
    ("customer escalation", [
        "the Acme account is threatening to churn over the latency regression",
        "we traced it to the new routing layer under burst load past 5K rps",
        "support promised a fix by Friday; eng thinks Tuesday is realistic",
        "we should offer a credit and a roadmap review call, not just an apology",
        "the renewal is 400K and the exec sponsor changed last month",
    ]),
    ("engineering standup", [
        "the cache-coherency bug only reproduces under co-resident memory pressure",
        "I rewrote the prefetch path; p50 dropped from 40ms to 0.4ms",
        "the page cache hit rate is sitting around 70 percent under the Zipf workload",
        "we still OOM on full Metal offload; partial offload with a trimmed prompt cache works",
        "the quant pass for the experts is done; router stays at 4-bit for safety",
    ]),
    ("sales pipeline review", [
        "pipeline coverage is 3.2x for the quarter but weighted is only 1.4x",
        "two deals slipped to next quarter on procurement, not on product",
        "the mid-market segment is closing 40 percent faster since the new demo flow",
        "we need two more SDRs to feed the enterprise team or the funnel dries up",
        "the competitor undercut us on the Globex deal; we won on the security review",
    ]),
    ("hiring debrief", [
        "the staff candidate was strong on systems but light on the Metal specifics",
        "we should make the offer but pair them with someone who knows the GPU path",
        "the panel was split on the design round; the take-home was excellent",
        "comp expectation is at the top of band; we can close the gap with equity",
        "let's move the second candidate to onsite and keep this one warm",
    ]),
]


def _make_transcript(sample_idx: int, min_chars: int) -> str:
    """Deterministic synthetic meeting transcript; varied by sample_idx (no RNG / no clock)."""
    topic_name, lines = _TOPICS[sample_idx % len(_TOPICS)]
    parts = [f"[Meeting: {topic_name} — session {sample_idx}]\n"]
    turn = sample_idx  # deterministic, index-varied
    while sum(len(p) for p in parts) < min_chars:
        spk = _SPEAKERS[turn % len(_SPEAKERS)]
        line = lines[turn % len(lines)]
        # Vary phrasing deterministically so repetition isn't a single pathological token run.
        filler = f" (point {turn}, ref {sample_idx}-{turn % 7})"
        parts.append(f"{spk}: {line}{filler}.\n")
        turn += 1
    return "".join(parts)


def _templated(tokenizer, transcript: str) -> tuple[str, int]:
    msgs = [{"role": "user", "content": SUMMARIZE_INSTRUCTION + transcript}]
    templated = tokenizer.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=False
    )
    return templated, len(tokenizer.encode(templated))


def build_transcript_prompt(tokenizer, target_tokens: int, sample_idx: int) -> tuple[str, int]:
    """Build a chat-templated summarization prompt of ~target_tokens, deterministically.

    CRITICAL (code-review R2-P1): we trim the TRANSCRIPT BODY and re-apply the chat template
    each time, so the assistant generation suffix (the "now answer" turn) is ALWAYS preserved
    at the end. Slicing the already-templated token sequence would strip that suffix and the
    model would just continue an unfinished user turn instead of summarizing — silently
    invalidating both the generated summary and the PPL gate.
    """
    # Grow the transcript body until the FULL templated prompt reaches the target.
    chars = target_tokens * 3
    transcript = _make_transcript(sample_idx, chars)
    _, ntok = _templated(tokenizer, transcript)
    while ntok < target_tokens:
        chars = int(chars * 1.3) + 256
        transcript = _make_transcript(sample_idx, chars)
        _, ntok = _templated(tokenizer, transcript)

    # Binary-search the transcript length (in chars) so the templated total lands near target.
    lo, hi = 0, len(transcript)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        _, n = _templated(tokenizer, transcript[:mid])
        if n <= target_tokens:
            lo = mid
        else:
            hi = mid - 1
    templated, actual = _templated(tokenizer, transcript[:lo])
    return templated, actual


# ---------------------------------------------------------------------------
# Analytic + direct KV-cache sizing
# ---------------------------------------------------------------------------


@dataclass
class ModelDims:
    n_layers: int
    n_kv_heads: int
    head_dim: int


def _model_dims(model) -> ModelDims:
    cfg = getattr(model, "args", None) or getattr(model, "config", None)
    # Qwen3 MoE config field names; fall back through common aliases.
    def g(*names, default=None):
        for n in names:
            v = getattr(cfg, n, None)
            if v is not None:
                return v
        return default

    n_layers = g("num_hidden_layers", "n_layers")
    n_kv = g("num_key_value_heads", "n_kv_heads", default=g("num_attention_heads", "n_heads"))
    hidden = g("hidden_size", "d_model")
    n_heads = g("num_attention_heads", "n_heads")
    head_dim = g("head_dim", default=(hidden // n_heads if hidden and n_heads else None))
    return ModelDims(int(n_layers), int(n_kv), int(head_dim))


def analytic_kv_bytes(dims: ModelDims, seq_len: int, kv_bits: Optional[int]) -> float:
    """Analytic KV-cache bytes. kv16 = fp16 K+V; kv4 = 4-bit + per-group fp16 scale+bias."""
    elems = 2 * dims.n_layers * dims.n_kv_heads * dims.head_dim * seq_len  # K + V elements
    if kv_bits is None:  # fp16
        return elems * 2
    # quantized: kv_bits per element + (scale+bias) fp16 per group of KV_GROUP_SIZE
    per_elem = kv_bits / 8.0
    groups = elems / KV_GROUP_SIZE
    return elems * per_elem + groups * 2 * 2  # 2 (scale,bias) * 2 bytes(fp16)


def direct_kv_bytes(prompt_cache) -> int:
    """Exact KV-cache bytes via each cache entry's .nbytes (KVCache or QuantizedKVCache)."""
    total = 0
    for c in prompt_cache:
        nb = getattr(c, "nbytes", None)
        if nb is not None:
            total += int(nb)
    return total


# ---------------------------------------------------------------------------
# Generation + teacher-forced scoring through the cached (possibly quantized) path
# ---------------------------------------------------------------------------


# Match mlx_lm.generate_step's default chunked prefill so quantization is interleaved between
# chunks exactly as the production path does (code-review R1): a single 15K-token forward pass
# would (a) apply quantization only once at the very end, and (b) allocate far larger prefill
# scratch than the chunked path the daily workload actually uses.
PREFILL_STEP_SIZE = 512


def _prefill_and_quantize(model, prompt_ids, kv_bits: Optional[int]):
    """Prefill prompt into a fresh prompt_cache using chunked prefill (mlx_lm semantics);
    quantize the cache progressively between chunks for kv4. Returns (cache, last_logits)."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.generate import maybe_quantize_kv_cache

    cache = make_prompt_cache(model)
    ids = mx.array(prompt_ids)
    n = ids.shape[0]
    last_logits = None
    i = 0
    while i < n:
        chunk = ids[i:i + PREFILL_STEP_SIZE][None]
        logits = model(chunk, cache=cache)
        # Interleave quantization between chunks (offset grows past quantized_kv_start=0).
        maybe_quantize_kv_cache(cache, QUANTIZED_KV_START, KV_GROUP_SIZE, kv_bits)
        mx.eval([c.state for c in cache])
        i += PREFILL_STEP_SIZE
        last_logits = logits[:, -1, :]
    mx.eval(last_logits)
    return cache, last_logits


def _assert_quantized(cache):
    from mlx_lm.models.cache import QuantizedKVCache

    bad = [i for i, c in enumerate(cache) if not isinstance(c, QuantizedKVCache)]
    return len(bad) == 0, bad


def greedy_generate(model, prompt_ids, kv_bits: Optional[int], max_tokens: int):
    """Greedy decode through the cached path. Returns (gen_ids, timing, cache)."""
    import time

    import mlx.core as mx

    t0 = time.perf_counter()
    cache, last_logits = _prefill_and_quantize(model, prompt_ids, kv_bits)
    tok = mx.argmax(last_logits, axis=-1)
    mx.eval(tok)
    ttft = time.perf_counter() - t0
    prefill_tps = len(prompt_ids) / ttft if ttft > 0 else 0.0

    gen = [int(tok.item())]
    t_dec = time.perf_counter()
    for _ in range(max_tokens - 1):
        logits = model(tok[None], cache=cache)
        tok = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(tok)
        gen.append(int(tok.item()))
    dec_elapsed = time.perf_counter() - t_dec
    decode_tps = (len(gen) - 1) / dec_elapsed if dec_elapsed > 0 else 0.0
    timing = {
        "ttft_s": round(ttft, 3),
        "prefill_tok_s": round(prefill_tps, 1),
        "decode_tok_s": round(decode_tps, 1),
    }
    return gen, timing, cache


def teacher_forced_nll(model, prompt_ids, target_ids, kv_bits: Optional[int]):
    """Score NLL on target_ids ONLY, conditioned on prompt via the (possibly quantized) cache.

    Returns (sum_nll, n_tokens, quantized_ok, bad_layers). Routes through the same incremental
    cached path generation uses, so the quantized K/V actually participate (plan-review R5-R2).
    """
    import mlx.core as mx
    import mlx.nn as nn

    cache, last_logits = _prefill_and_quantize(model, prompt_ids, kv_bits)
    quantized_ok, bad = (True, [])
    if kv_bits is not None:
        quantized_ok, bad = _assert_quantized(cache)

    # last_logits predicts the FIRST target token; then feed each target token to get the next.
    total_nll = 0.0
    logits = last_logits  # predicts target_ids[0]
    for i, tgt in enumerate(target_ids):
        logp = nn.log_softmax(logits, axis=-1)
        total_nll += -float(logp[0, tgt].item())
        if i < len(target_ids) - 1:
            tok = mx.array([tgt])
            out = model(tok[None], cache=cache)
            logits = out[:, -1, :]
            mx.eval(logits)
    return total_nll, len(target_ids), quantized_ok, bad


def rouge_l(ref_ids, hyp_ids) -> float:
    """ROUGE-L (LCS-based F1) over token id sequences. Divergence trigger, not a gate."""
    if not ref_ids or not hyp_ids:
        return 0.0
    n, m = len(ref_ids), len(hyp_ids)
    # LCS length via DP (token sequences are <=256, cheap).
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            dp[j] = prev + 1 if ref_ids[i - 1] == hyp_ids[j - 1] else max(dp[j], dp[j - 1])
            prev = tmp
    lcs = dp[m]
    if lcs == 0:
        return 0.0
    prec, rec = lcs / m, lcs / n
    return 2 * prec * rec / (prec + rec)


# ---------------------------------------------------------------------------
# Memory preflight (two-stage, no double-count — plan-review R5-R1)
# ---------------------------------------------------------------------------


def preflight(dims: ModelDims, seq_len: int, kv_bits: Optional[int], model_loaded: bool) -> dict:
    kv_gb = analytic_kv_bytes(dims, seq_len + MAX_TOKENS, kv_bits) / (1024**3)
    incremental = kv_gb + SCRATCH_GB + SAFETY_GB
    total = WEIGHTS_GB + incremental
    avail = get_available_memory_gb()
    if not model_loaded:
        ok = total <= USABLE_CAP_GB
        return {"stage": "pre_load", "analytic_kv_gb": round(kv_gb, 3),
                "total_required_gb": round(total, 2), "usable_cap_gb": USABLE_CAP_GB,
                "available_gb": round(avail, 2), "ok": ok}
    ok = incremental <= avail
    return {"stage": "post_load", "analytic_kv_gb": round(kv_gb, 3),
            "incremental_required_gb": round(incremental, 2),
            "available_gb": round(avail, 2), "ok": ok}


# ---------------------------------------------------------------------------
# Phase A — smoke / compatibility gate
# ---------------------------------------------------------------------------


def cmd_smoke(args):
    import mlx.core as mx
    from mlx_lm import load

    LOGDIR.mkdir(parents=True, exist_ok=True)
    avail = get_available_memory_gb()
    print(f"[smoke] available memory: {avail:.2f} GB")
    if avail < WEIGHTS_GB + 1.5:
        print(f"[smoke] ABORT: < {WEIGHTS_GB + 1.5:.1f} GB free — quiesce the machine "
              f"(quit Cursor/Chrome/extra Claude procs) per the machine-contention blocker.")
        log_experiment(
            "h9_e2_smoke", "h9_e2_kv_workload",
            {"model": MODEL}, {"status": "aborted_low_memory", "available_gb": round(avail, 2)},
            status="aborted",
        )
        return 1

    print(f"[smoke] loading {MODEL} ...")
    model, tokenizer = load(MODEL)
    dims = _model_dims(model)
    print(f"[smoke] dims: n_layers={dims.n_layers} n_kv_heads={dims.n_kv_heads} "
          f"head_dim={dims.head_dim}")
    loaded_gb, _ = tree_footprint_gb(psutil.Process().pid)
    print(f"[smoke] loaded footprint: {loaded_gb:.2f} GB")

    # kv4 compatibility gate: short + medium prompt, confirm slope differs (plan-review R3-medium).
    results = {}
    for label, ntok in [("short", 256), ("medium", 2048)]:
        prompt, actual = build_transcript_prompt(tokenizer, ntok, 0)
        pid = tokenizer.encode(prompt)
        for kvb, key in [(None, "kv16"), (4, "kv4")]:
            cache, _ = _prefill_and_quantize(model, pid, kvb)
            direct = direct_kv_bytes(cache)
            ana = analytic_kv_bytes(dims, actual, kvb)
            qok, bad = (True, []) if kvb is None else _assert_quantized(cache)
            results[f"{label}_{key}"] = {
                "seq": actual, "direct_kv_mb": round(direct / 1e6, 2),
                "analytic_kv_mb": round(ana / 1e6, 2),
                "quantized_ok": qok, "bad_layers": bad,
            }
            print(f"  {label}/{key}: seq={actual} direct={direct/1e6:.1f}MB "
                  f"analytic={ana/1e6:.1f}MB quantized_ok={qok}")
            del cache
            gc.collect()
            mx.clear_cache()

    # Slope check: kv4 cache must grow ~3.5x slower than kv16 between short and medium.
    d16 = results["medium_kv16"]["direct_kv_mb"] - results["short_kv16"]["direct_kv_mb"]
    d4 = results["medium_kv4"]["direct_kv_mb"] - results["short_kv4"]["direct_kv_mb"]
    slope_ratio = (d16 / d4) if d4 > 0 else float("inf")
    compat_ok = results["medium_kv4"]["quantized_ok"] and slope_ratio >= 2.5
    print(f"[smoke] kv16 slope={d16:.1f}MB kv4 slope={d4:.1f}MB ratio={slope_ratio:.2f} "
          f"=> compat_ok={compat_ok}")

    status = "completed" if compat_ok else "infra_fail_kv_quant_unsupported"
    log_experiment(
        "h9_e2_smoke", "h9_e2_kv_workload",
        {"model": MODEL, "kv_group_size": KV_GROUP_SIZE,
         "quantized_kv_start": QUANTIZED_KV_START,
         "n_layers": dims.n_layers, "n_kv_heads": dims.n_kv_heads, "head_dim": dims.head_dim},
        {"loaded_footprint_gb": round(loaded_gb, 2), "slope_ratio_kv16_over_kv4": round(slope_ratio, 2),
         "compat_ok": compat_ok, "cells": results,
         "cache_hit_rate": None,
         "cache_hit_rate_reason": "not applicable: no expert/page cache in MLX KV quant arm",
         "gpu_memory_mb": round(mx.get_peak_memory() / 1e6, 1)},
        status=status,
    )
    if not compat_ok:
        print("[smoke] INFRA FAIL: kv4 path did not change the memory slope as expected.")
        return 1
    print("[smoke] PASS — kv4 quantization is engaged and the measurement path works.")
    return 0


# ---------------------------------------------------------------------------
# Phase B/C — the sweep
# ---------------------------------------------------------------------------


def _summarize_once(model, tokenizer, prompt_ids, kv_bits):
    """Generate a greedy summary; return (gen_ids, summary_text, timing, kv_direct, kv_analytic, qok)."""
    import mlx.core as mx

    dims = _model_dims(model)
    mx.reset_peak_memory()
    base_active = mx.get_active_memory()
    gen, timing, cache = greedy_generate(model, prompt_ids, kv_bits, MAX_TOKENS)
    mx.eval([c.state for c in cache])
    cache_active = mx.get_active_memory()
    direct = direct_kv_bytes(cache)
    analytic = analytic_kv_bytes(dims, len(prompt_ids) + len(gen), kv_bits)
    qok, _ = (True, []) if kv_bits is None else _assert_quantized(cache)
    steady_delta = max(0, cache_active - base_active)
    text = tokenizer.decode(gen)
    del cache
    gc.collect()
    mx.clear_cache()
    return {
        "gen_ids": gen, "summary": text, "timing": timing,
        "kv_direct_bytes": direct, "kv_analytic_bytes": analytic,
        "kv_steady_delta_bytes": steady_delta, "quantized_ok": qok,
        "peak_gpu_mb": round(mx.get_peak_memory() / 1e6, 1),
    }


def cmd_run(args):
    from mlx_lm import load

    LOGDIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    env = get_environment_info()

    lengths = args.lengths or LENGTHS
    n_samples = args.samples or SAMPLES_PER_LENGTH

    avail = get_available_memory_gb()
    print(f"[run] available memory: {avail:.2f} GB")
    if avail < WEIGHTS_GB + 1.5:
        print(f"[run] ABORT: quiesce the machine first (need > {WEIGHTS_GB + 1.5:.1f} GB free).")
        return 1

    print(f"[run] loading {MODEL} ...")
    model, tokenizer = load(MODEL)
    dims = _model_dims(model)
    print(f"[run] dims: {dims}")

    per_length_verdicts = {}

    for L in lengths:
        print(f"\n========== context length {L} ==========")
        # Pre-load budget is informational here (model already loaded); run post-load check.
        pf16 = preflight(dims, L, None, model_loaded=True)
        pf4 = preflight(dims, L, 4, model_loaded=True)
        print(f"[preflight L={L}] kv16 {pf16}  | kv4 {pf4}")

        samples_kv16, samples_kv4 = [], []
        kv16_skipped = not pf16["ok"]
        kv4_skipped = not pf4["ok"]

        for s in range(n_samples):
            prompt, actual = build_transcript_prompt(tokenizer, L, s)
            (TRANSCRIPT_DIR / f"L{L}_s{s}.txt").write_text(prompt)
            prompt_ids = tokenizer.encode(prompt)
            print(f"  [L={L} s={s}] prompt tokens={actual}")

            # kv16 first — its summary is the pinned scoring + ROUGE target.
            r16 = None
            if not kv16_skipped:
                try:
                    r16 = _summarize_once(model, tokenizer, prompt_ids, None)
                except Exception as e:  # OOM or runtime
                    print(f"    kv16 FAILED at L={L}: {e}")
                    kv16_skipped = True
            r4 = None
            if not kv4_skipped:
                try:
                    r4 = _summarize_once(model, tokenizer, prompt_ids, 4)
                except Exception as e:
                    print(f"    kv4 FAILED at L={L}: {e}")
                    kv4_skipped = True

            # Quality: teacher-force the kv16 summary under both configs (summary-only ΔPPL).
            ppl16 = ppl4 = None
            quantized_ok = None
            rouge = None
            if r16 is not None:
                target = r16["gen_ids"]
                nll16, n16, _, _ = teacher_forced_nll(model, prompt_ids, target, None)
                ppl16 = math.exp(nll16 / n16)
                if r4 is not None:
                    nll4, n4, quantized_ok, bad = teacher_forced_nll(model, prompt_ids, target, 4)
                    ppl4 = math.exp(nll4 / n4)
                    rouge = rouge_l(r16["gen_ids"], r4["gen_ids"])
                    (SUMMARY_DIR / f"L{L}_s{s}_kv16.txt").write_text(r16["summary"])
                    (SUMMARY_DIR / f"L{L}_s{s}_kv4.txt").write_text(r4["summary"])

            if r16 is not None:
                samples_kv16.append({"sample_idx": s, "seq": actual, "ppl": ppl16, **r16["timing"],
                                     "kv_direct_bytes": r16["kv_direct_bytes"],
                                     "kv_analytic_bytes": r16["kv_analytic_bytes"],
                                     "peak_gpu_mb": r16["peak_gpu_mb"]})
            if r4 is not None:
                samples_kv4.append({"sample_idx": s, "seq": actual, "ppl": ppl4, **r4["timing"],
                                    "kv_direct_bytes": r4["kv_direct_bytes"],
                                    "kv_analytic_bytes": r4["kv_analytic_bytes"],
                                    "peak_gpu_mb": r4["peak_gpu_mb"],
                                    "quantized_ok": quantized_ok, "rouge_l_vs_kv16": rouge})
            if ppl16 is not None and ppl4 is not None:
                d = abs(ppl4 - ppl16) / ppl16
                print(f"    ppl16={ppl16:.4f} ppl4={ppl4:.4f} Δrel={d:.4%} "
                      f"rougeL={rouge:.3f} quantized_ok={quantized_ok}")

        verdict = _aggregate_and_log(L, dims, samples_kv16, samples_kv4,
                                     kv16_skipped, kv4_skipped, pf16, pf4, env)
        per_length_verdicts[L] = verdict

    _final_verdict(per_length_verdicts, env)
    return 0


def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def _aggregate_and_log(L, dims, kv16, kv4, kv16_skipped, kv4_skipped, pf16, pf4, env) -> dict:
    # Use the REALIZED median seq length (code-review R5): tokenizer round-trip can make the
    # actual prompt differ from L by a few tokens, and the cache also holds MAX_TOKENS decoded.
    def realized_seq(samples):
        s = _median([x["seq"] for x in samples])
        return int(s) + MAX_TOKENS if s is not None else (L + MAX_TOKENS)
    seq16, seq4 = realized_seq(kv16), realized_seq(kv4)

    def kv_gb(samples, kvb, seq):
        if samples:
            return _median([s["kv_direct_bytes"] for s in samples]) / (1024**3)
        return analytic_kv_bytes(dims, seq, kvb) / (1024**3)

    kv16_gb = kv_gb(kv16, None, seq16)
    kv4_gb = kv_gb(kv4, 4, seq4)

    # .nbytes vs analytic hard validity gate (per config, when measured) — analytic at the
    # realized seq so the comparison is apples-to-apples.
    def validity(samples, kvb, seq):
        if not samples:
            return {"measured": False}
        direct = _median([s["kv_direct_bytes"] for s in samples])
        ana = analytic_kv_bytes(dims, seq, kvb)
        diverge = abs(direct - ana) / ana
        return {"measured": True, "direct_bytes": int(direct), "analytic_bytes": int(ana),
                "divergence": round(diverge, 4), "valid": diverge <= NBYTES_ANALYTIC_TOL}
    v16, v4 = validity(kv16, None, seq16), validity(kv4, 4, seq4)
    memory_inconclusive = (v16.get("measured") and not v16["valid"]) or \
                          (v4.get("measured") and not v4["valid"])

    # per-GB ratio uses each config's own realized seq (they're near-identical by construction).
    per_gb_16 = seq16 / kv16_gb
    per_gb_4 = seq4 / kv4_gb
    per_gb_ratio = per_gb_4 / per_gb_16

    # Quality gate: summary-only ΔPPL on samples PAIRED BY sample_idx (code-review R4 — a
    # mid-sweep failure in one config must not silently mis-pair the positional lists).
    kv4_by_idx = {s["sample_idx"]: s for s in kv4}
    deltas = []
    for a in kv16:
        b = kv4_by_idx.get(a["sample_idx"])
        if b and a["ppl"] and b["ppl"]:
            deltas.append(abs(b["ppl"] - a["ppl"]) / a["ppl"])
    mean_dppl = statistics.mean(deltas) if deltas else None
    rouges = [s.get("rouge_l_vs_kv16") for s in kv4 if s.get("rouge_l_vs_kv16") is not None]
    mean_rouge = statistics.mean(rouges) if rouges else None

    # Speed flag.
    dec16, dec4 = _median([s["decode_tok_s"] for s in kv16]), _median([s["decode_tok_s"] for s in kv4])
    pre16, pre4 = _median([s["prefill_tok_s"] for s in kv16]), _median([s["prefill_tok_s"] for s in kv4])
    speed_flag = bool(dec16 and dec4 and dec4 < dec16 * (1 - SPEED_FLAG_REL))

    # Gate evaluation (plan.md verdict classes).
    quality_pass = (mean_dppl is not None and mean_dppl <= PPL_GATE_REL)
    quality_status = "pass" if quality_pass else ("unverified" if kv16_skipped else "fail")
    # per-GB is meaningless (pure analytic, nothing ran) when BOTH configs were skipped —
    # surface None so _final_verdict doesn't read a spurious pass (code-review R3).
    both_skipped = kv16_skipped and kv4_skipped
    if both_skipped:
        pergb_pass = None
    elif memory_inconclusive:
        # .nbytes vs analytic diverged > tolerance — the memory measurement is invalid, so the
        # per-GB gate cannot PASS on it (code-review R2-P2).
        pergb_pass = False
    else:
        pergb_pass = per_gb_ratio >= PERGB_GATE
    no_oom = not kv4_skipped
    kv16_oom_path = kv16_skipped and not kv4_skipped  # PASS-with-asterisk case

    verdict = {
        "length": L,
        "quality_status": quality_status,
        "mean_dppl_rel": round(mean_dppl, 5) if mean_dppl is not None else None,
        "mean_rouge_l": round(mean_rouge, 4) if mean_rouge is not None else None,
        "rouge_below_review_floor": bool(mean_rouge is not None and mean_rouge < ROUGE_REVIEW_FLOOR),
        "per_gb_ratio": round(per_gb_ratio, 3),
        "per_gb_pass": pergb_pass,
        "kv16_gb": round(kv16_gb, 4), "kv4_gb": round(kv4_gb, 4),
        "no_oom_at_length": no_oom,
        "kv16_oom_kv4_ok": kv16_oom_path,
        "memory_inconclusive": memory_inconclusive,
        "speed_flag_kv4_slower": speed_flag,
    }

    config = {
        "model": MODEL, "runtime": "mlx_lm", "tier": "engine_offhours",
        "context_length": L, "kv_group_size": KV_GROUP_SIZE,
        "quantized_kv_start": QUANTIZED_KV_START, "max_tokens": MAX_TOKENS,
        "n_kv16_samples": len(kv16), "n_kv4_samples": len(kv4),
        "n_layers": dims.n_layers, "n_kv_heads": dims.n_kv_heads, "head_dim": dims.head_dim,
        "preflight_kv16": pf16, "preflight_kv4": pf4,
    }
    results = {
        "mem_method": "phys_footprint+nbytes",
        "verdict": verdict,
        "kv16_samples": kv16, "kv4_samples": kv4,
        "kv16_validity": v16, "kv4_validity": v4,
        "median_decode_tok_s": {"kv16": dec16, "kv4": dec4},
        "median_prefill_tok_s": {"kv16": pre16, "kv4": pre4},
        "gpu_memory_mb": _median([s["peak_gpu_mb"] for s in (kv4 or kv16)]),
        "cache_hit_rate": None,
        "cache_hit_rate_reason": "not applicable: no expert/page cache in MLX KV quant arm",
        "perplexity": {"kv16_mean": _median([s["ppl"] for s in kv16]),
                       "kv4_mean": _median([s["ppl"] for s in kv4])},
        "kv16_skipped": kv16_skipped, "kv4_skipped": kv4_skipped,
    }
    status = "completed"
    if kv4_skipped and kv16_skipped:
        status = "skipped_oom_preflight"
    log_experiment(f"h9_e2_kv_workload_L{L}", "h9_e2_kv_workload", config, results,
                   status=status, env=env)
    print(f"[verdict L={L}] {verdict}")
    return verdict


def _final_verdict(per_length, env):
    lengths = sorted(per_length)
    quals = {L: per_length[L]["quality_status"] for L in lengths}
    pergbs = {L: per_length[L]["per_gb_pass"] for L in lengths}
    all_quality = all(q == "pass" for q in quals.values())
    all_pergb = all(pergbs.values())
    # For the asterisk path, a None per-GB (both configs skipped at a length) shouldn't drag
    # an otherwise-clean result down to PARTIAL (code-review R3-low) — judge on the lengths
    # that actually produced a per-GB number.
    pergb_where_measured = all(v for v in pergbs.values() if v is not None)
    any_unverified = any(q == "unverified" for q in quals.values())

    if all_quality and all_pergb:
        overall = "PASS"
    elif pergb_where_measured and any_unverified and not any(q == "fail" for q in quals.values()):
        overall = "PASS_WITH_ASTERISK"  # e.g. 15K kv16 OOM, quality proven through 12K
    elif quals.get(lengths[0]) == "pass" and quals.get(lengths[1] if len(lengths) > 1 else lengths[0]) == "pass":
        overall = "PARTIAL"
    else:
        overall = "FAIL"

    summary = {
        "overall_verdict": overall,
        "scope": "24GB off-hours MLX engine tier; NOT the 16GB daily-driver budget; "
                 "NOT vllm-mlx served (follow-up). Arm (b) llama.cpp deferred (E1b host blocker).",
        "per_length": per_length,
        "quality_by_length": quals,
        "per_gb_pass_by_length": pergbs,
    }
    log_experiment("h9_e2_kv_workload_SUMMARY", "h9_e2_kv_workload",
                   {"model": MODEL, "lengths": lengths}, summary,
                   status="completed", env=env)
    print(f"\n========== OVERALL VERDICT: {overall} ==========")
    print(summary)


def main():
    ap = argparse.ArgumentParser(description="H9-E2 KV quantization at workload context lengths")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    r = sub.add_parser("run")
    r.add_argument("--lengths", type=int, nargs="*", default=None)
    r.add_argument("--samples", type=int, default=None)
    args = ap.parse_args()
    if args.cmd == "smoke":
        return cmd_smoke(args)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
