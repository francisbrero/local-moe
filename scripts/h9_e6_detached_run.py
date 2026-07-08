"""
h9_e6_detached_run.py — bundled detached quiesced measurement run (issue #44).

Three experiments (E1b/#41, E2/#36, E4/#38) produced ZERO measured numbers, all for the
same root cause: memory contention from the orchestrating IDE/agent session itself. On this
24 GB machine, with Cursor + the agent + Chrome live, free RAM hovers at 9.5-10.6 GB while
the MLX-wired 30B needs ~18 GB. The only lever that closes the gap is quitting the IDE, which
cannot happen mid-session. This script is the detached remedy: a HARD preflight gate (>=19 GB
free) that must be satisfied from a bare terminal, then all three experiments run in one shot
with per-phase memory hygiene, each logging >=1 number to experiments.jsonl.

=============================================================================================
# HOW TO RUN DETACHED
# 1. Quit Cursor / VS Code and ALL IDE + agent windows (not just minimize).
# 2. Quit Chrome and any other large apps.
# 3. Open Terminal.app (or another bare terminal, e.g. cmux — NOT an IDE-integrated terminal).
# 4. Run:
#      cd /Users/francis/Documents/MadKudu/local-moe
#      uv run python scripts/h9_e6_detached_run.py 2>&1 | tee dev/active/issue-44-h9-e6-detached-run/logs/e6_run.log
# 5. Results land in experiments.jsonl (records prefixed "h9_e6_"). The tee'd log captures
#    the console trace for overnight runs.
#
# The preflight will ABORT (exit 2) if free RAM < 19 GB. That abort is by design: an agent
# attempting this from inside an IDE session will see the abort and the reason. Use --force
# ONLY to exercise the harness path when you cannot quiesce (it still logs + warns loudly and
# will very likely OOM on the first MLX load).
#
# Execution order: Phase 0 (preflight) -> E4 (idle MLX baseline, FIRST to protect the "idle"
# claim) -> E1b (partial-offload decode) -> E2 (KV sweep: MLX kv4 + llama.cpp q8_0) -> summary.
=============================================================================================

Usage:
  uv run python scripts/h9_e6_detached_run.py                 # full run
  uv run python scripts/h9_e6_detached_run.py --e1b-ngl 24    # override partial-offload layers
  uv run python scripts/h9_e6_detached_run.py --skip e1b e2   # run a subset
  uv run python scripts/h9_e6_detached_run.py --dry-run       # preflight + capability probes only
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

SCRIPTS_DIR = Path(__file__).parent
REPO = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiment_utils import (  # noqa: E402
    get_available_memory_gb,
    get_environment_info,
    get_vm_stat,
    log_experiment,
    vm_stat_delta,
)

# E1b primitives we reuse directly (NOT h9_e1b_coresidency.run / _compute_gates — those tear
# the server down before the tool-call harness can run and gate mmap on the wrong metric).
import h9_e1b_coresidency as e1b  # noqa: E402
from h9_e1_baseline import served_generate_timed  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_FREE_GB = 19.0            # hard preflight gate (issue #44 launch precondition)
COOLDOWN_S = 8.0              # let the compressor/page-cache settle between phases

# E1b partial-offload defaults (48 hidden layers; -ngl 24 = 50%). --cache-ram trimmed to dodge
# the documented GPU-OOM (30B + prompt cache + KV > 18186 MiB GPU budget at full offload).
DEFAULT_E1B_NGL = 24
DEFAULT_CACHE_RAM_MIB = 512
DEFAULT_E1B_MINUTES = 30.0

# E2 llama.cpp KV-quant arm ports (distinct from E1b's 8124 so a not-yet-released port from the
# E1b phase can't block bind).
PORT_E2_F16 = 8125
PORT_E2_Q8 = 8126

CORPUS_DIR = REPO / "dev" / "active" / "issue-44-h9-e6-detached-run" / "corpus"
LOGDIR = REPO / "dev" / "active" / "issue-44-h9-e6-detached-run" / "logs"

# Track whether any MLX work happened so cooldown knows to clear the Metal cache.
_mlx_touched = False


# ---------------------------------------------------------------------------
# Preflight + inter-phase hygiene
# ---------------------------------------------------------------------------


def preflight(force: bool) -> bool:
    """Hard gate. Returns True to proceed. On failure logs an aborted record and (unless
    --force) the caller exits non-zero."""
    avail = get_available_memory_gb()
    swap = e1b.swap_used_gb()
    print("=" * 80)
    print("H9-E6 DETACHED RUN — PREFLIGHT")
    print("=" * 80)
    print(f"  available RAM : {avail:.2f} GB   (need >= {MIN_FREE_GB:.0f} GB)")
    print(f"  swap used     : {swap} GB")
    if avail >= MIN_FREE_GB:
        print("  PREFLIGHT PASS — machine is quiesced enough to proceed.\n")
        return True

    shortfall = MIN_FREE_GB - avail
    print("\n  PREFLIGHT ABORT")
    print(f"  Only {avail:.2f} GB free; short by {shortfall:.2f} GB.")
    print("  The MLX-wired 30B needs ~18 GB and this bundled run needs a clean >=19 GB slate.")
    print("  To fix:")
    print("    1. Quit Cursor / VS Code and every IDE + agent window (not just minimize).")
    print("    2. Quit Chrome and other large apps.")
    print("    3. Re-run from a bare terminal (Terminal.app / cmux).")
    print("  If the agent driving this IS inside the IDE, that session is part of the working")
    print("  set it's trying to clear — it must be closed. This abort is the enforcement.\n")
    log_experiment(
        experiment_name="h9_e6_preflight",
        phase="h9_e6",
        config={"min_free_gb": MIN_FREE_GB, "force": force},
        results={"available_gb": round(avail, 2), "shortfall_gb": round(shortfall, 2),
                 "swap_used_gb": swap},
        status="aborted",
        env=get_environment_info(),
    )
    if force:
        print("  --force set: continuing anyway (expect OOM on first MLX load).\n")
        return True
    return False


def cooldown_and_gate(min_free_gb: float, label: str) -> dict:
    """Force teardown to settle, then re-read available RAM. Returns a state dict; caller
    decides whether to skip the next phase if `ok` is False."""
    global _mlx_touched
    gc.collect()
    if _mlx_touched:
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass
    time.sleep(COOLDOWN_S)
    avail = get_available_memory_gb()
    swap = e1b.swap_used_gb()
    ok = avail >= min_free_gb
    print(f"[cooldown->{label}] avail={avail:.2f} GB (need {min_free_gb:.0f}) swap={swap} "
          f"-> {'OK' if ok else 'LOW-MEM SKIP'}")
    return {"available_gb": round(avail, 2), "swap_used_gb": swap,
            "min_free_gb": min_free_gb, "ok": ok}


def _nearest_rank_p95(xs: list[float]) -> float:
    s = sorted(xs)
    if not s:
        return 0.0
    return s[min(math.ceil(0.95 * len(s)) - 1, len(s) - 1)]


# ---------------------------------------------------------------------------
# Phase C (runs FIRST): E4 — idle-machine MLX baseline
# ---------------------------------------------------------------------------


def phase_e4() -> dict:
    """Idle MLX baseline: decode/prefill tok/s + phys_footprint peak + 20-case tool-call.

    Runs the standalone h9_e1_baseline (perf) and h9_e1_harness (tool-call, engine tier) as
    SUBPROCESSES so each acquires + fully releases the ~18 GB MLX model in its own address
    space — guaranteeing a clean teardown before the next phase (the inter-phase hygiene
    requirement). Both scripts self-log to experiments.jsonl; we additionally emit a
    consolidated h9_e6_e4 record referencing them.
    """
    global _mlx_touched
    _mlx_touched = True
    LOGDIR.mkdir(parents=True, exist_ok=True)

    perf_out = LOGDIR / "e4_perf.log"
    print("[E4] running MLX perf baseline (decode/prefill/phys) via h9_e1_baseline ...")
    perf_rc = _run_subprocess(
        [sys.executable, str(SCRIPTS_DIR / "h9_e1_baseline.py"), "--runtime", "engine"],
        perf_out,
    )

    tool_out = LOGDIR / "e4_toolcall.json"
    print("[E4] running 20-case tool-call harness (engine tier) ...")
    tool_rc = _run_subprocess(
        [sys.executable, str(SCRIPTS_DIR / "h9_e1_harness.py"),
         "--runtime", "engine", "--out", str(tool_out)],
        LOGDIR / "e4_toolcall.log",
    )
    # h9_e1_harness --out writes {"summary": {...}, "cases": [...]}; the rate is nested.
    tool_doc = _read_json(tool_out)
    tool_rate = ((tool_doc or {}).get("summary") or {}).get("semantic_pass_rate")

    result = {
        "ran_first": True,
        "idle_state": {"available_gb": round(get_available_memory_gb(), 2),
                       "swap_used_gb": e1b.swap_used_gb()},
        "perf_subprocess_rc": perf_rc,
        "perf_log": str(perf_out),
        "toolcall_subprocess_rc": tool_rc,
        "toolcall_semantic_pass_rate": tool_rate,
        "note": "decode/prefill/phys_footprint recorded by h9_e1_baseline (phase perf_engine); "
                "this record consolidates + adds the tool-call rate. Gates: decode>=20, "
                "prefill>=300@8k, tool>=90%, phys<=21GB (see h9_e1_baseline record for numbers).",
    }
    # E4 requires BOTH the perf number and the tool-call number to count as producing numbers.
    status = "completed" if (perf_rc == 0 and tool_rate is not None) else "subprocess_error"
    log_experiment("h9_e6_e4", "h9_e6", {"runtime": "engine"}, result, status=status,
                   env=get_environment_info())
    return {"status": status, **result}


# ---------------------------------------------------------------------------
# Phase A (runs after E4): E1b — partial-offload decode
# ---------------------------------------------------------------------------


def _start_e1b_server(ngl: int, cache_ram_mib: int, port: int, log_path: Path):
    """Copy E1b's llama-server arg list (do NOT call e1b.start_server — its signature has no
    cache_ram param and pins PORT), append --cache-ram to trim the prompt cache, and Popen."""
    args = [
        e1b.LLAMA_SERVER,
        "-m", e1b.MODEL_PATH,
        "--port", str(port),
        "--ctx-size", str(e1b.CTX_SIZE),
        "--batch-size", str(e1b.BATCH),
        "--ubatch-size", str(e1b.UBATCH),
        "--parallel", str(e1b.PARALLEL),
        "-ngl", str(ngl),
        "--cache-ram", str(cache_ram_mib),
        "--jinja",
        "--metrics",
        # mmap ENABLED (default): do NOT pass --no-mmap or --mlock.
    ]
    f = open(log_path, "w")
    try:
        proc = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT)
    except BaseException:
        f.close()
        raise
    proc._log_fh = f
    return proc


def phase_e1b(ngl: int, cache_ram_mib: int, minutes: float) -> dict:
    """Partial-offload decode under 8 GB ballast, then tool-call harness against the SAME live
    server. mmap footprint gated on ri_resident_size (NOT phys_footprint). Orchestrated at
    low-level primitives inside try/finally so the server survives long enough for the harness."""
    LOGDIR.mkdir(parents=True, exist_ok=True)
    port = e1b.PORT  # 8124
    base_url = f"http://127.0.0.1:{port}"
    server_log = LOGDIR / f"e1b_server_ngl{ngl}.log"

    ballast = e1b.Ballast()
    ballast_info: dict = {}
    proc = None
    vm_before = get_vm_stat()

    try:
        print(f"[E1b] starting llama-server -ngl {ngl} --cache-ram {cache_ram_mib} (mmap on)")
        proc = _start_e1b_server(ngl, cache_ram_mib, port, server_log)
        load_t0 = time.perf_counter()
        healthy = e1b.wait_healthy(base_url, proc, timeout_s=900)
        load_s = time.perf_counter() - load_t0
        if not healthy:
            print(f"[E1b] LOAD FAILED after {load_s:.0f}s")
            result = {"verdict": "load_failed", "ngl": ngl, "cache_ram_mib": cache_ram_mib,
                      "load_seconds": round(load_s, 1),
                      "buffer_sizes": e1b.parse_buffer_sizes(server_log)}
            log_experiment("h9_e6_e1b", "h9_e6", {"ngl": ngl}, result, status="load_failed",
                           env=get_environment_info())
            return {"status": "load_failed", **result}

        print(f"[E1b] healthy after {load_s:.0f}s; allocating 8 GB ballast (load-then-ballast)")
        buffers = e1b.parse_buffer_sizes(server_log)
        ballast_info = ballast.allocate()

        # Warmup so all samples start warm.
        try:
            served_generate_timed(base_url, [{"role": "user", "content": "Say hello."}], 16)
        except Exception as exc:
            print(f"[E1b] warmup errored: {exc}")

        # Sustained generation + sampling loop (adapted from e1b.run's inner loop).
        samples: list[dict] = []
        decodes: list[dict] = []
        root = proc.pid
        start = time.perf_counter()
        end_t = start + minutes * 60
        next_sample = start
        print(f"[E1b] sustained loop {minutes} min, sampling every {e1b.SAMPLE_INTERVAL_S}s")
        while time.perf_counter() < end_t:
            try:
                timed = served_generate_timed(
                    base_url, [{"role": "user", "content": e1b.DECODE_PROMPT}],
                    max_tokens=e1b.DECODE_MAX_TOKENS,
                )
                decodes.append({"ttft_ms": timed.get("ttft_ms"),
                                "decode_tok_s": timed.get("decode_tok_s")})
            except Exception as exc:
                print(f"[E1b] decode errored: {exc}")
                decodes.append({"ttft_ms": None, "decode_tok_s": None})

            now = time.perf_counter()
            if now >= next_sample:
                c = e1b.tree_counters(root)
                samples.append({
                    "t": now - start,
                    "phys_gb": c["phys_footprint_b"] / 1024**3,
                    "resident_gb": c["resident_b"] / 1024**3,
                    "diskread_b": c["diskio_bytesread_b"],
                    "swap_gb": e1b.swap_used_gb(),
                    "avail_gb": get_available_memory_gb(),
                    "mem_method": c["mem_method"],
                })
                ballast.retouch()
                s = samples[-1]
                print(f"  [{int(s['t'])}s] resident={s['resident_gb']:.2f}GB "
                      f"phys={s['phys_gb']:.2f}GB avail={s['avail_gb']:.2f}GB swap={s['swap_gb']}")
                next_sample = now + e1b.SAMPLE_INTERVAL_S

        vmd = vm_stat_delta(vm_before, get_vm_stat())

        # Tool-call harness against the STILL-ALIVE server (import, don't shell out).
        print("[E1b] running 20-case tool-call harness (served, same server) ...")
        tool_rate = _run_toolcall_served(base_url)

        result = _e1b_gates(ngl, cache_ram_mib, samples, decodes, vmd, ballast_info,
                            buffers, load_s, tool_rate, minutes)
        log_experiment("h9_e6_e1b", "h9_e6", {"ngl": ngl, "cache_ram_mib": cache_ram_mib,
                       "minutes": minutes}, result, status="completed",
                       env=get_environment_info())
        return {"status": "completed", **result}
    finally:
        if proc is not None:
            _kill(proc)
        ballast.cpu = None
        ballast.mlx = None
        gc.collect()


def _e1b_gates(ngl, cache_ram_mib, samples, decodes, vmd, ballast_info, buffers, load_s,
               tool_rate, minutes) -> dict:
    resident = [s["resident_gb"] for s in samples] or [0.0]
    phys = [s["phys_gb"] for s in samples] or [0.0]
    res_p95 = _nearest_rank_p95(resident)
    res_max = max(resident)
    tps = [d["decode_tok_s"] for d in decodes if d["decode_tok_s"] is not None]
    ttfts = [d["ttft_ms"] for d in decodes if d["ttft_ms"] is not None]
    decode_med = statistics.median(tps) if tps else None
    ttft_p50_s = (statistics.median(ttfts) / 1000.0) if ttfts else None
    hours = (samples[-1]["t"] - samples[0]["t"]) / 3600.0 if len(samples) >= 2 else minutes / 60.0
    pageout_mb_hr = (vmd["pageout_delta_mb"] / hours) if hours > 0 else None

    # Full unified-memory budget: ri_resident_size alone can pass while total pressure thrashes.
    ballast_total = ballast_info.get("ballast_total_gb", 8.0)
    budget_line = {
        "ballast_gb": ballast_total,
        "mmap_working_set_gb": round(res_max, 2),      # ri_resident_size (authoritative for mmap)
        "phys_footprint_gb_reference": round(max(phys), 2),
        "buffer_sizes_raw": buffers.get("raw", []),    # Metal model buffer / KV / prompt cache
        "headroom_gb": 1.0,
        "note": "Metal buffer + KV + prompt-cache come from buffer_sizes_raw (parsed from the "
                "llama.cpp load banner); sum + ballast + mmap working set + 1 GB headroom should "
                "fit within ~20 GB usable. Higher --e1b-ngl grows the Metal buffer.",
    }
    est_total = ballast_total + res_max + 1.0  # + parsed buffers (in raw); conservative floor
    budget_exceeded = est_total > 20.0

    gates = {
        "decode_ge_12": (decode_med is not None and decode_med >= 12.0),
        "ttft_p50_le_10": (ttft_p50_s is not None and ttft_p50_s <= 10.0),
        "pageout_lt_200mb_hr": (pageout_mb_hr is not None and pageout_mb_hr < 200.0),
        "resident_le_8gb": (res_p95 <= 8.0 and res_max <= 8.0),  # mmap gate on ri_resident_size
        "toolcall_ge_90pct": (tool_rate is not None and tool_rate >= 0.90),
    }
    return {
        "ngl": ngl, "cache_ram_mib": cache_ram_mib, "minutes": minutes,
        "load_seconds": round(load_s, 1),
        "decode_tok_s_median": decode_med,
        "ttft_p50_s": ttft_p50_s,
        "pageout_mb_hr": round(pageout_mb_hr, 1) if pageout_mb_hr is not None else None,
        "resident_gb_p95": round(res_p95, 2), "resident_gb_max": round(res_max, 2),
        "mem_method": samples[-1]["mem_method"] if samples else "phys_footprint",
        "toolcall_semantic_pass_rate": tool_rate,
        "unified_memory_budget": budget_line,
        "budget_exceeded": budget_exceeded,
        "ballast": ballast_info,
        "gates": gates,
        "n_samples": len(samples), "n_decodes": len(decodes),
    }


def _run_toolcall_served(base_url: str) -> Optional[float]:
    """Run the imported 20-case harness against a live OpenAI-compatible server."""
    try:
        import h9_e1_harness as harness

        results = harness.run_served(base_url)
        summary = harness.summarize(results)
        return summary.get("semantic_pass_rate")
    except Exception as exc:
        print(f"[E1b] tool-call harness errored: {exc}")
        return None


# ---------------------------------------------------------------------------
# Phase B (runs last): E2 — KV quantization sweep
# ---------------------------------------------------------------------------


def phase_e2_mlx() -> dict:
    """Arm 1: MLX kv_bits=4 sweep. Delegate to the standalone h9_e2_kv_workload as a subprocess
    (it owns its own two-stage preflight + logging). Consolidate into an h9_e6_e2_mlx record."""
    global _mlx_touched
    _mlx_touched = True
    LOGDIR.mkdir(parents=True, exist_ok=True)
    out = LOGDIR / "e2_mlx.log"
    print("[E2-MLX] running kv4 sweep via h9_e2_kv_workload run ...")
    rc = _run_subprocess(
        [sys.executable, str(SCRIPTS_DIR / "h9_e2_kv_workload.py"), "run"],
        out,
    )
    result = {
        "subprocess_rc": rc,
        "log": str(out),
        "note": "MLX kv4 vs kv16 quality (paired dNLL) + effective-context/GB recorded by "
                "h9_e2_kv_workload (records h9_e2_kv_workload_L*/SUMMARY). Gate: dNLL within "
                "harness threshold AND >=1.5x ctx/GB vs FP16.",
    }
    status = "completed" if rc == 0 else "subprocess_error"
    log_experiment("h9_e6_e2_mlx", "h9_e6", {"arm": "mlx_kv4"}, result, status=status,
                   env=get_environment_info())
    return {"status": status, **result}


def _build_e2_corpus(n_docs: int = 3, target_tokens: int = 12000) -> list[dict]:
    """Deterministic fixed corpus (pinned sample_idx) saved + SHA256-logged so both llama.cpp
    arms score identical inputs and the comparison is reproducible without a prior E2 run.
    Uses a tokenizer-only load to size prompts; no MLX weights are loaded (avoids materializing
    the full 30B just before the llama.cpp+ballast arm, which would OOM)."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    import h9_e2_kv_workload as e2
    from h9_e1_baseline import _load_tokenizer_only  # downloads tokenizer files only, no weights

    tokenizer = _load_tokenizer_only()
    if tokenizer is None:
        raise RuntimeError("tokenizer-only load failed; cannot build E2 corpus without weights")
    corpus = []
    for i in range(n_docs):
        prompt, ntok = e2.build_transcript_prompt(tokenizer, target_tokens, sample_idx=i)
        path = CORPUS_DIR / f"transcript_{i}.txt"
        path.write_text(prompt)
        sha = hashlib.sha256(prompt.encode()).hexdigest()
        corpus.append({"idx": i, "path": str(path), "sha256": sha, "n_tokens": ntok})
        print(f"[E2-llama] corpus doc {i}: {ntok} tok, sha256={sha[:12]} -> {path.name}")
    return corpus


def _probe_input_logprobs(base_url: str, prompt: str) -> dict:
    """Verify llama-server /completion returns logprobs covering INPUT tokens (array length ==
    prompt token count), not just sampled tokens. Known limitation: llama-server's n_probs
    covers sampled positions only, so this is expected to report unsupported -> quality
    degrades to None."""
    import requests

    try:
        r = requests.post(f"{base_url}/completion",
                          json={"prompt": prompt, "n_predict": 1, "n_probs": 1,
                                "temperature": 0.0},
                          timeout=60)
        r.raise_for_status()
        data = r.json()
        probs = data.get("completion_probabilities") or data.get("logprobs") or []
        # We would need per-input-token coverage; llama-server returns only sampled-token probs.
        prompt_tokens = data.get("tokens_evaluated")
        covers_input = (isinstance(probs, list) and prompt_tokens is not None
                        and len(probs) >= prompt_tokens)
        return {"supported": bool(covers_input),
                "n_probs_returned": len(probs) if isinstance(probs, list) else 0,
                "prompt_tokens": prompt_tokens}
    except Exception as exc:
        return {"supported": False, "error": str(exc)}


def _run_e2_llama_arm(kv_type: str, port: int, corpus: list[dict]) -> dict:
    """Launch one llama-server arm (-fa -ctk <kv_type>) under 8 GB ballast; measure pageout +
    memory + quality-probe. Returns an arm result dict."""
    LOGDIR.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{port}"
    server_log = LOGDIR / f"e2_llama_{kv_type}.log"
    ballast = e1b.Ballast()
    proc = None
    vm_before = get_vm_stat()
    try:
        args = [
            e1b.LLAMA_SERVER, "-m", e1b.MODEL_PATH,
            "--port", str(port),
            "--ctx-size", str(e1b.CTX_SIZE),
            "--batch-size", str(e1b.BATCH),
            "--ubatch-size", str(e1b.UBATCH),
            "--parallel", str(e1b.PARALLEL),
            "-ngl", str(DEFAULT_E1B_NGL),
            "--cache-ram", str(DEFAULT_CACHE_RAM_MIB),
            "-fa",                 # flash attention: REQUIRED for KV quantization
            "-ctk", kv_type,       # KV cache type: q8_0 (arm) or f16 (baseline)
            "--jinja", "--metrics",
        ]
        f = open(server_log, "w")
        try:
            proc = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT)
        except BaseException:
            f.close()  # don't leak the handle if Popen fails (e.g. binary missing)
            raise
        proc._log_fh = f
        print(f"[E2-llama:{kv_type}] starting llama-server -fa -ctk {kv_type} on {port}")
        if not e1b.wait_healthy(base_url, proc, timeout_s=900):
            return {"kv_type": kv_type, "verdict": "load_failed",
                    "buffer_sizes": e1b.parse_buffer_sizes(server_log)}
        ballast_info = ballast.allocate()

        probe_text = Path(corpus[0]["path"]).read_text()[:2000] if corpus else ""
        quality_probe = _probe_input_logprobs(base_url, probe_text)

        # Summarization workload over the fixed corpus; sample memory each doc.
        samples = []
        root = proc.pid
        for doc in corpus:
            prompt = Path(doc["path"]).read_text()
            try:
                served_generate_timed(base_url, [{"role": "user", "content": prompt}],
                                      max_tokens=256)
            except Exception as exc:
                print(f"[E2-llama:{kv_type}] gen errored: {exc}")
            c = e1b.tree_counters(root)
            samples.append({"resident_gb": c["resident_b"] / 1024**3,
                            "phys_gb": c["phys_footprint_b"] / 1024**3})
            ballast.retouch()
        vmd = vm_stat_delta(vm_before, get_vm_stat())
        res_max = max((s["resident_gb"] for s in samples), default=0.0)
        return {
            "kv_type": kv_type,
            "buffer_sizes": e1b.parse_buffer_sizes(server_log),
            "quality_method": "served_input_logprobs" if quality_probe.get("supported")
                              else "unavailable",
            "quality_probe": quality_probe,
            "quality_na_reason": None if quality_probe.get("supported")
                                 else "llama-server REST exposes sampled-token logprobs only, "
                                      "not input-token; teacher-forced NLL not obtainable.",
            "pageout_delta_mb": vmd["pageout_delta_mb"],
            "resident_gb_max": round(res_max, 2),
            "ballast": ballast_info,
            "n_docs": len(corpus),
        }
    finally:
        if proc is not None:
            _kill(proc)
        ballast.cpu = None
        ballast.mlx = None
        gc.collect()


def phase_e2_llama(force: bool = False) -> dict:
    """Arm 2 (NEW): llama.cpp -ctk q8_0 vs -ctk f16 baseline, each under 8 GB ballast.
    Quality via input-token logprob probe (expected unavailable -> None); pageout gated
    absolute (<200 MB/hr) AND relative (q8_0 <= f16 + 200 MB/hr)."""
    corpus = _build_e2_corpus()  # tokenizer-only; no MLX weights materialized
    cd = cooldown_and_gate(MIN_FREE_GB - 4.0, "e2-llama-arms")  # 8 GB ballast needs ~15 GB free
    if not cd["ok"] and not force:
        result = {"status": "skipped_lowmem", "cooldown": cd, "corpus": corpus}
        log_experiment("h9_e6_e2_llamacpp", "h9_e6", {"arm": "llamacpp_q8_0_vs_f16"},
                       result, status="skipped_lowmem", env=get_environment_info())
        return result

    f16 = _run_e2_llama_arm("f16", PORT_E2_F16, corpus)
    time.sleep(COOLDOWN_S)  # let the f16 server + ballast release before the q8_0 arm
    gc.collect()
    q8 = _run_e2_llama_arm("q8_0", PORT_E2_Q8, corpus)

    po_q8 = q8.get("pageout_delta_mb")
    po_f16 = f16.get("pageout_delta_mb")
    # These are cumulative deltas over each arm's short workload, not per-hour; report both and
    # the gate as absolute<200MB (proxy) AND relative(q8<=f16+200MB). Documented as coarse.
    gate_abs = (po_q8 is not None and po_q8 < 200.0)
    gate_rel = (po_q8 is not None and po_f16 is not None and po_q8 <= po_f16 + 200.0)
    result = {
        "corpus": corpus,
        "arm_f16": f16,
        "arm_q8_0": q8,
        "pageout_gate_absolute_lt200mb": gate_abs,
        "pageout_gate_relative_le_f16_plus200mb": gate_rel,
        "quality_method": q8.get("quality_method"),
        "note": "Pageout deltas are per-workload cumulative (vm_stat), not per-hour; treat as a "
                "coarse regression proxy. Quality sub-gate is None when input-token logprobs are "
                "unavailable (the expected llama-server case).",
    }
    status = "completed"
    if f16.get("verdict") == "load_failed" or q8.get("verdict") == "load_failed":
        status = "load_failed"
    log_experiment("h9_e6_e2_llamacpp", "h9_e6", {"arm": "llamacpp_q8_0_vs_f16",
                   "cooldown_gate": cd}, result, status=status, env=get_environment_info())
    return {"status": status, **result}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kill(proc: subprocess.Popen):
    try:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    except Exception:
        pass
    fh = getattr(proc, "_log_fh", None)
    if fh is not None:
        try:
            fh.close()
        except Exception:
            pass


def _run_subprocess(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO))
    return proc.returncode


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="H9-E6 bundled detached measurement run (#44)")
    ap.add_argument("--e1b-ngl", type=int, default=DEFAULT_E1B_NGL,
                    help="partial-offload layers for E1b (default 24 = 50%% of 48).")
    ap.add_argument("--cache-ram", type=int, default=DEFAULT_CACHE_RAM_MIB,
                    help="llama.cpp --cache-ram MiB (trim prompt cache to dodge GPU-OOM).")
    ap.add_argument("--e1b-minutes", type=float, default=DEFAULT_E1B_MINUTES)
    ap.add_argument("--skip", nargs="*", default=[], choices=["e4", "e1b", "e2mlx", "e2llama"],
                    help="phases to skip.")
    ap.add_argument("--force", action="store_true",
                    help="bypass the <19GB preflight abort (still logs; expect OOM).")
    ap.add_argument("--dry-run", action="store_true",
                    help="preflight only; do not run any experiment.")
    args = ap.parse_args()

    if not preflight(args.force):
        sys.exit(2)
    if args.dry_run:
        print("[dry-run] preflight complete; not running experiments.")
        return

    summary = {"order": ["e4", "e1b", "e2mlx", "e2llama"], "phases": {}}

    # Phase C (E4) FIRST — genuine idle baseline.
    if "e4" not in args.skip:
        summary["phases"]["e4"] = phase_e4()

    # Phase A (E1b).
    if "e1b" not in args.skip:
        cd = cooldown_and_gate(MIN_FREE_GB - 4.0, "e1b")  # 8 GB ballast: ~15 GB free suffices
        if cd["ok"] or args.force:
            summary["phases"]["e1b"] = phase_e1b(args.e1b_ngl, args.cache_ram, args.e1b_minutes)
        else:
            summary["phases"]["e1b"] = {"status": "skipped_lowmem", "cooldown": cd}
            log_experiment("h9_e6_e1b", "h9_e6", {}, {"cooldown": cd},
                           status="skipped_lowmem", env=get_environment_info())

    # Phase B Arm 1 (E2 MLX) — needs the full ~18 GB idle budget.
    if "e2mlx" not in args.skip:
        cd = cooldown_and_gate(MIN_FREE_GB, "e2mlx")
        if cd["ok"] or args.force:
            summary["phases"]["e2mlx"] = phase_e2_mlx()
        else:
            summary["phases"]["e2mlx"] = {"status": "skipped_lowmem", "cooldown": cd}
            log_experiment("h9_e6_e2_mlx", "h9_e6", {}, {"cooldown": cd},
                           status="skipped_lowmem", env=get_environment_info())

    # Phase B Arm 2 (E2 llama.cpp).
    if "e2llama" not in args.skip:
        summary["phases"]["e2llama"] = phase_e2_llama(force=args.force)

    # Phase D — overall verdict: did every arm produce >=1 number?
    produced = {k: v.get("status") for k, v in summary["phases"].items()}
    # Vacuous-true guard: with every phase skipped there are no numbers, so the gate is False.
    all_ran = bool(produced) and all(s == "completed" for s in produced.values())
    summary["overall_gate_all_produced_numbers"] = all_ran
    summary["phase_statuses"] = produced
    print("\n" + "=" * 80)
    print("H9-E6 SUMMARY")
    print("=" * 80)
    for k, s in produced.items():
        print(f"  {k:>8}: {s}")
    print(f"  overall (all produced numbers): {all_ran}")
    log_experiment("h9_e6_SUMMARY", "h9_e6", {"skipped": args.skip}, summary,
                   status="completed" if all_ran else "partial", env=get_environment_info())


if __name__ == "__main__":
    main()
