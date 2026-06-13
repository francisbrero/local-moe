"""
h9_e1_baseline.py — H9-E1 Tier 1 performance baseline (issue #35).

Measures, with the pinned protocol from plan.md Phase C:
  - decode tok/s (cold + warm median, generated-tokens-only)
  - TTFT (reported separately)
  - prefill tok/s @ 8K and @ 16K context
  - per-condition memory: phys_footprint of the serving process tree (gate metric),
    MLX Metal active/peak (engine tier only), headroom vs the 19-21 GB practical budget

Runtimes:
  served  — vllm-mlx OpenAI-compatible endpoint (gate_tier=served). Memory via process-tree
            phys_footprint of the server PID; timing via streaming SSE deltas.
  engine  — in-process mlx_lm (gate_tier=engine). Memory via own phys_footprint + MLX counters.

Gates (issue #35): decode >= 20 tok/s, peak footprint <= 17 GB (16K steady-state row).
Tool-call gate is evaluated by h9_e1_harness.py.

Usage:
    uv run python scripts/h9_e1_baseline.py --runtime served --port 8123 --server-pid <pid>
    uv run python scripts/h9_e1_baseline.py --runtime engine
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import psutil

from experiment_utils import get_environment_info, log_experiment

MODEL = "mlx-community/Qwen3-30B-A3B-4bit"
PRACTICAL_BUDGET_GB = 21.0  # ~19-21 GB usable on the 24GB M4 Pro; use the upper bound for headroom
GATE_DECODE_TOKS = 20.0
GATE_FOOTPRINT_GB = 17.0
DECODE_TOKENS = 96  # tokens per warm decode run; lower => shorter runs under memory contention

# ---------------------------------------------------------------------------
# macOS phys_footprint (proc_pid_rusage RUSAGE_INFO_V2 -> ri_phys_footprint)
# ---------------------------------------------------------------------------

_RUSAGE_INFO_V2 = 2


class _RUsageInfoV2(ctypes.Structure):
    # struct rusage_info_v2 — only fields up to ri_phys_footprint matter; pad the rest.
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
    ]


_libc = ctypes.CDLL("libc.dylib", use_errno=True)


def phys_footprint_bytes(pid: int) -> Optional[int]:
    """macOS phys_footprint for a pid via proc_pid_rusage; None if unavailable."""
    info = _RUsageInfoV2()
    rc = _libc.proc_pid_rusage(
        ctypes.c_int(pid), ctypes.c_int(_RUSAGE_INFO_V2), ctypes.byref(info)
    )
    if rc != 0:
        return None
    return int(info.ri_phys_footprint)


def tree_pids(root_pid: int) -> list[int]:
    try:
        p = psutil.Process(root_pid)
        return [root_pid] + [c.pid for c in p.children(recursive=True)]
    except psutil.Error:
        return [root_pid]


def tree_footprint_gb(root_pid: int) -> tuple[float, str]:
    """Sum phys_footprint (fallback RSS) across the process tree. Returns (gb, method)."""
    total = 0
    method = "phys_footprint"
    for pid in tree_pids(root_pid):
        pf = phys_footprint_bytes(pid)
        if pf is None:
            try:
                pf = psutil.Process(pid).memory_info().rss
                method = "rss"
            except psutil.Error:
                pf = 0
        total += pf
    return total / (1024**3), method


# ---------------------------------------------------------------------------
# Memory sampler (background thread; peak per condition)
# ---------------------------------------------------------------------------


class MemSampler:
    def __init__(self, root_pid: int, engine_mode: bool, interval_s: float = 0.2):
        self.root_pid = root_pid
        self.engine_mode = engine_mode
        self.interval = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.peak_footprint_gb = 0.0
        self.peak_metal_gb = 0.0
        self.mem_method = "phys_footprint"

    def _run(self):
        while not self._stop.is_set():
            gb, method = tree_footprint_gb(self.root_pid)
            self.peak_footprint_gb = max(self.peak_footprint_gb, gb)
            self.mem_method = method
            if self.engine_mode:
                try:
                    import mlx.core as mx

                    self.peak_metal_gb = max(
                        self.peak_metal_gb, mx.get_peak_memory() / (1024**3)
                    )
                except Exception:
                    pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


# ---------------------------------------------------------------------------
# Served-tier timing (streaming SSE)
# ---------------------------------------------------------------------------


def served_generate_timed(base_url: str, messages: list[dict], max_tokens: int) -> dict:
    """Stream a completion; return TTFT, decode tok/s (gen-only), token counts.

    decode tok/s excludes prefill: rate = (n_chunks-1)/(t_last - t_first_token).
    """
    import requests

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
    }
    t0 = time.perf_counter()
    t_first = None
    t_last = None
    n_tokens = 0
    with requests.post(
        f"{base_url}/v1/chat/completions", json=payload, stream=True, timeout=600
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            s = line.decode("utf-8")
            if not s.startswith("data: "):
                continue
            data = s[len("data: ") :]
            if data.strip() == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = obj.get("choices", [{}])[0].get("delta", {})
            if delta.get("content") or delta.get("tool_calls"):
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                n_tokens += 1
    ttft_ms = (t_first - t0) * 1000 if t_first else None
    decode_toks = (
        (n_tokens - 1) / (t_last - t_first)
        if (t_first and t_last and t_last > t_first and n_tokens > 1)
        else None
    )
    return {
        "ttft_ms": round(ttft_ms, 1) if ttft_ms else None,
        "decode_tok_s": round(decode_toks, 2) if decode_toks else None,
        "gen_tokens": n_tokens,
        "timing_method": "streaming_sse",
    }


# ---------------------------------------------------------------------------
# Engine-tier timing (in-process mlx_lm)
# ---------------------------------------------------------------------------


def engine_generate_timed(model, tokenizer, prompt: str, max_tokens: int) -> dict:
    """Use mlx_lm.stream_generate, which reports prompt_tps / generation_tps natively.

    decode tok/s = generation_tps (decode only, excludes prefill).
    prefill tok/s = prompt_tps. TTFT measured as wall-clock to first streamed token.
    """
    from mlx_lm import stream_generate

    t0 = time.perf_counter()
    t_first = None
    last = None
    for resp in stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens):
        if t_first is None:
            t_first = time.perf_counter()
        last = resp
    ttft_ms = (t_first - t0) * 1000 if t_first else None
    return {
        "ttft_ms": round(ttft_ms, 1) if ttft_ms else None,
        "decode_tok_s": round(last.generation_tps, 2) if last else None,
        "prefill_tok_s": round(last.prompt_tps, 2) if last else None,
        "n_prompt": last.prompt_tokens if last else 0,
        "gen_tokens": last.generation_tokens if last else 0,
        "peak_memory_gb": round(last.peak_memory, 3) if last and last.peak_memory else None,
        "timing_method": "stream_generate_tps",
    }


# ---------------------------------------------------------------------------
# Prompt synthesis for context lengths
# ---------------------------------------------------------------------------

_TRANSCRIPT = (
    "Alice: Let's review the Q3 roadmap. We shipped onboarding and fixed three P1 bugs. "
    "Bob: Latency improved 30 percent after the cache rewrite, but memory use rose slightly. "
    "Carol: Customer churn dropped after the pricing change; support tickets are up ten percent. "
    "Dave: We should prioritize the billing migration before the renewal cycle in July. "
)


def _load_tokenizer_only():
    """Load just the tokenizer for the model (no weights), for served-mode token counts.

    Returns an object with .encode(str)->list. None if it can't be loaded (then served
    prefill tok/s falls back to a char heuristic / null).
    """
    try:
        from pathlib import Path

        from huggingface_hub import snapshot_download
        from mlx_lm.tokenizer_utils import load as load_tokenizer

        path = snapshot_download(
            MODEL, allow_patterns=["*.json", "tokenizer*", "*.txt", "*.model"]
        )
        return load_tokenizer(Path(path))
    except Exception as e:
        print(f"  (served tokenizer-only load failed: {e}; using char heuristic)")
        return None


def build_prompt_of_length(tokenizer, target_tokens: int) -> tuple[str, int]:
    """Repeat transcript text until tokenized length is within +-5% of target."""
    base = _TRANSCRIPT
    text = base
    while len(tokenizer.encode(text)) < target_tokens:
        text += base
    # Trim back toward target by token slicing.
    toks = tokenizer.encode(text)[:target_tokens]
    text = tokenizer.decode(toks)
    actual = len(tokenizer.encode(text))
    return text, actual


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def run_engine(n_warm: int) -> dict:
    import mlx.core as mx
    from mlx_lm import load

    pid = psutil.Process().pid
    print(f"  Loading {MODEL} in-process (pid {pid})...")
    model, tokenizer = load(MODEL)

    idle_gb, _ = tree_footprint_gb(pid)
    print(f"  idle_loaded footprint: {idle_gb:.2f} GB")

    decode_prompt = "Write a short paragraph about distributed systems."
    decode_msgs_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": decode_prompt}],
        add_generation_prompt=True, tokenize=False,
    )

    # cold
    cold = engine_generate_timed(model, tokenizer, decode_msgs_text, max_tokens=64)
    print(f"  cold decode: {cold['decode_tok_s']} tok/s, TTFT {cold['ttft_ms']} ms")

    # warm decode (median over n_warm)
    warm_rates = []
    warm_ttfts = []
    with MemSampler(pid, engine_mode=True) as ms_decode:
        for i in range(n_warm):
            r = engine_generate_timed(model, tokenizer, decode_msgs_text, max_tokens=DECODE_TOKENS)
            warm_rates.append(r["decode_tok_s"])
            warm_ttfts.append(r["ttft_ms"])
            print(f"    warm[{i}] decode {r['decode_tok_s']} tok/s")
    warm_median = statistics.median([x for x in warm_rates if x])

    # prefill @ 8K and @ 16K
    prefill = {}
    mem_rows = {"idle_loaded": round(idle_gb, 2),
                "decode_short": round(ms_decode.peak_footprint_gb, 2)}
    metal_rows = {"decode_short": round(ms_decode.peak_metal_gb, 2)}
    for ctx_name, target in (("8k", 8192), ("16k", 16384)):
        text, actual = build_prompt_of_length(tokenizer, target)
        msgs_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": text + "\n\nSummarize the above in one sentence."}],
            add_generation_prompt=True, tokenize=False,
        )
        with MemSampler(pid, engine_mode=True) as ms:
            r = engine_generate_timed(model, tokenizer, msgs_text, max_tokens=8)
        prefill[ctx_name] = {
            "target_tokens": target, "actual_prompt_tokens": r["n_prompt"],
            "prefill_tok_s": r["prefill_tok_s"], "ttft_ms": r["ttft_ms"],
        }
        mem_rows[f"prefill_decode_{ctx_name}"] = round(ms.peak_footprint_gb, 2)
        metal_rows[f"prefill_decode_{ctx_name}"] = round(ms.peak_metal_gb, 2)
        print(f"  prefill {ctx_name}: {r['prefill_tok_s']} tok/s "
              f"({r['n_prompt']} tok), footprint {ms.peak_footprint_gb:.2f} GB")

    return _assemble(
        runtime="mlx_lm", gate_tier="engine", gating=True,
        cold=cold, warm_median=warm_median, warm_rates=warm_rates, warm_ttfts=warm_ttfts,
        prefill=prefill, mem_rows=mem_rows, metal_rows=metal_rows,
        mem_method="phys_footprint",
    )


def run_served(base_url: str, server_pid: int, n_warm: int) -> dict:
    idle_gb, mem_method = tree_footprint_gb(server_pid)
    print(f"  served idle_loaded footprint (pid-tree {server_pid}): {idle_gb:.2f} GB ({mem_method})")

    decode_msgs = [{"role": "user", "content": "Write a short paragraph about distributed systems."}]

    cold = served_generate_timed(base_url, decode_msgs, max_tokens=64)
    print(f"  cold decode: {cold['decode_tok_s']} tok/s, TTFT {cold['ttft_ms']} ms")

    warm_rates, warm_ttfts = [], []
    with MemSampler(server_pid, engine_mode=False) as ms_decode:
        for i in range(n_warm):
            r = served_generate_timed(base_url, decode_msgs, max_tokens=256)
            warm_rates.append(r["decode_tok_s"])
            warm_ttfts.append(r["ttft_ms"])
            print(f"    warm[{i}] decode {r['decode_tok_s']} tok/s")
    warm_median = statistics.median([x for x in warm_rates if x])

    # Load only the tokenizer (no model weights) to count prompt tokens for served prefill tok/s.
    served_tokenizer = _load_tokenizer_only()

    prefill = {}
    mem_rows = {"idle_loaded": round(idle_gb, 2),
                "decode_short": round(ms_decode.peak_footprint_gb, 2)}
    for ctx_name, target in (("8k", 8192), ("16k", 16384)):
        if served_tokenizer is not None:
            text, _actual = build_prompt_of_length(served_tokenizer, target)
        else:
            text = _TRANSCRIPT * max(1, target // 60)  # rough char fallback
        content = text + "\n\nSummarize the above in one sentence."
        msgs = [{"role": "user", "content": content}]
        with MemSampler(server_pid, engine_mode=False) as ms:
            r = served_generate_timed(base_url, msgs, max_tokens=8)
        # prompt token count: local tokenizer if available (approx; excludes chat-template overhead).
        prompt_tokens = len(served_tokenizer.encode(content)) if served_tokenizer else None
        ttft_s = (r["ttft_ms"] / 1000.0) if r["ttft_ms"] else None
        prefill_toks = (
            round(prompt_tokens / ttft_s, 2)
            if (prompt_tokens and ttft_s and ttft_s > 0)
            else None
        )
        prefill[ctx_name] = {
            "target_tokens": target,
            "prompt_tokens": prompt_tokens,
            "ttft_ms": r["ttft_ms"],
            "prefill_tok_s": prefill_toks,
            "timing_method": ("prompt_tokens/TTFT (TTFT≈prefill at cold KV; approximate)"
                              if prefill_toks is not None
                              else "unavailable_served (no local tokenizer or TTFT)"),
        }
        mem_rows[f"prefill_decode_{ctx_name}"] = round(ms.peak_footprint_gb, 2)
        print(f"  prefill {ctx_name}: {prefill_toks} tok/s "
              f"({prompt_tokens} tok, TTFT {r['ttft_ms']} ms), footprint {ms.peak_footprint_gb:.2f} GB")

    return _assemble(
        runtime="vllm-mlx", gate_tier="served", gating=True,
        cold=cold, warm_median=warm_median, warm_rates=warm_rates, warm_ttfts=warm_ttfts,
        prefill=prefill, mem_rows=mem_rows,
        metal_rows={"_note": "metal counters unavailable in served mode (separate process)"},
        mem_method=mem_method,
    )


def _assemble(runtime, gate_tier, gating, cold, warm_median, warm_rates, warm_ttfts,
              prefill, mem_rows, metal_rows, mem_method) -> dict:
    peak_16k = mem_rows.get("prefill_decode_16k", max(mem_rows.values()))
    return {
        "runtime": runtime,
        "gate_tier": gate_tier,
        "gating": gating,
        "status": "completed",
        "decode": {
            "cold_tok_s": cold["decode_tok_s"],
            "warm_median_tok_s": round(warm_median, 2) if warm_median else None,
            "warm_rates": warm_rates,
            "cold_ttft_ms": cold["ttft_ms"],
            "warm_ttft_ms": warm_ttfts,
        },
        "prefill": prefill,
        "memory_gb": mem_rows,
        "metal_gb": metal_rows,
        "mem_method": mem_method,
        "gate_footprint_row": "prefill_decode_16k",
        "peak_footprint_16k_gb": peak_16k,
        "headroom_vs_practical_gb": round(PRACTICAL_BUDGET_GB - peak_16k, 2),
        "gates": {
            "decode_ge_20": (warm_median or 0) >= GATE_DECODE_TOKS,
            "footprint_le_17": peak_16k <= GATE_FOOTPRINT_GB,
            "operational_on_24gb": peak_16k <= PRACTICAL_BUDGET_GB,
        },
        # repo-convention fields kept consistent across experiments:
        "perplexity": None,
        "perplexity_na_reason": "quality represented by tool-call rate; 4-bit PPL covered by H7 (PR #20)",
        "cache_hit_rate": None,
        "cache_hit_rate_na_reason": "resident model, no SSD expert streaming; cache hits measured in H0 (PR #16)",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", choices=["served", "engine"], required=True)
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--server-pid", type=int, default=None,
                    help="Root PID of the vllm-mlx server process tree (served mode).")
    ap.add_argument("--n-warm", type=int, default=5)
    ap.add_argument("--no-log", action="store_true", help="Skip writing to experiments.jsonl.")
    args = ap.parse_args()

    if args.runtime == "served":
        if not args.server_pid:
            raise SystemExit("--server-pid is required for served mode (process-tree memory).")
        results = run_served(f"http://localhost:{args.port}", args.server_pid, args.n_warm)
        phase = "perf_served"
    else:
        results = run_engine(args.n_warm)
        phase = "perf_engine"

    print("\n=== H9-E1 baseline results ===")
    print(json.dumps(results, indent=2))

    if not args.no_log:
        log_experiment(
            experiment_name="h9_e1_baseline",
            phase=phase,
            config={"model": MODEL, "runtime": results["runtime"],
                    "gate_tier": results["gate_tier"], "gating": results["gating"],
                    "n_warm": args.n_warm},
            results=results,
            status=results["status"],
            env=get_environment_info(),
        )


if __name__ == "__main__":
    main()
