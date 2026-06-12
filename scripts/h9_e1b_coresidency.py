"""
h9_e1b_coresidency.py — llama.cpp mmap co-residency probe for Qwen3-30B-A3B (issue #41).

Decides the shape of the H9 concurrent tier: does the 30B model in llama.cpp mmap mode
survive co-residency under an 8 GB competing working set, where MLX-wired thrashed in E1
(#35: 16.55 GB wired phys_footprint, swap-collapse, 4x load failures)?

Method (see dev/active/issue-41-h9-e1b-mmap-coresidency/plan.md):
  - Launch llama-server with mmap enabled (default), --jinja, --metrics, pinned ctx/batch.
  - Measure two mandatory -ngl configs: 999 (full Metal offload) and 0 (CPU/mmap-preserved);
    bounded partial-offload search (25/50/75% + <=2 bisection) only if both extremes fail.
  - Allocate EXACTLY 8 GB fixed ballast (4 GB CPU random-filled + 4 GB Metal), periodically
    re-touched so the macOS compressor cannot quietly purge it.
  - Sample every 30 s over a >=30 min sustained-generation window: process-tree phys_footprint,
    ri_resident_size, ri_diskio_bytesread (per-process mmap churn), pageins, vm_stat, swap.
  - Two load orders: B1 (load-then-ballast, primary 30-min gate) and B2 (ballast-then-load,
    load-under-pressure, >=5 min sustain).

Gates (all 6 must pass; see plan.md):
  decode >= 12 tok/s | p50 TTFT <= 10 s | pageouts < 200 MB/hr | phys_footprint p95 AND max
  <= 8 GB | disk-read churn < 50 MB/s | tool-call success >= 90% (run via h9_e1_harness.py).

Usage:
  uv run python scripts/h9_e1b_coresidency.py calibrate
  uv run python scripts/h9_e1b_coresidency.py run --ngl 999 --order B1 --minutes 30
  uv run python scripts/h9_e1b_coresidency.py run --ngl 0   --order B2 --minutes 5
"""

from __future__ import annotations

import argparse
import ctypes
import os
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import psutil

# Reuse the authoritative phys_footprint collector + SSE timing from the E1 baseline.
from h9_e1_baseline import (  # noqa: E402
    _RUsageInfoV2,
    _RUSAGE_INFO_V2,
    _libc,
    phys_footprint_bytes,
    served_generate_timed,
    tree_pids,
)
from experiment_utils import (  # noqa: E402
    allocate_cpu_ballast,
    allocate_mlx_ballast,
    get_available_memory_gb,
    get_environment_info,
    get_vm_stat,
    log_experiment,
    vm_stat_delta,
)

LLAMA_SERVER = "/Users/francis/.docker/bin/inference/llama-server"
MODEL_PATH = str(
    Path(__file__).parent.parent / "models" / "gguf" / "Qwen3-30B-A3B-Q4_K_M.gguf"
)
MODEL_REPO = "Qwen/Qwen3-30B-A3B-GGUF"
MODEL_FILE = "Qwen3-30B-A3B-Q4_K_M.gguf"
MODEL_SHA256 = "0d003f6662faee786ed5da3e31b29c978de5ae5d275c8794c606a7f3c01aa8f5"
LOGDIR = Path(__file__).parent.parent / "dev" / "active" / "issue-41-h9-e1b-mmap-coresidency" / "logs"

# Pinned configuration (plan.md "Pinned configuration") — every value logged.
CTX_SIZE = 8192
BATCH = 2048
UBATCH = 512
PARALLEL = 1
PORT = 8124
BALLAST_CPU_GB = 4.0
BALLAST_MLX_GB = 4.0
SAMPLE_INTERVAL_S = 30.0
DECODE_MAX_TOKENS = 256
# A ~512-token prompt for the sustained decode probe (pinned for run-to-run comparability).
DECODE_PROMPT = (
    "You are an operations assistant. Carefully read the following context and then write a "
    "detailed, well-structured summary covering every salient point, open question, and next "
    "action. Be thorough and concrete.\n\n"
) + ("The quarterly planning thread contains many overlapping decisions, owners, and dates. " * 24)


# ---------------------------------------------------------------------------
# Per-process RUSAGE_INFO_V2 counters (pageins, resident, disk-read) — process tree sums
# ---------------------------------------------------------------------------


def _rusage_v2(pid: int) -> Optional[_RUsageInfoV2]:
    info = _RUsageInfoV2()
    rc = _libc.proc_pid_rusage(
        ctypes.c_int(pid), ctypes.c_int(_RUSAGE_INFO_V2), ctypes.byref(info)
    )
    return info if rc == 0 else None


def tree_counters(root_pid: int) -> dict:
    """Sum phys_footprint / resident / pageins / diskio_bytesread across the process tree."""
    pf = res = pagein = diskread = 0
    method = "phys_footprint"
    for pid in tree_pids(root_pid):
        info = _rusage_v2(pid)
        if info is not None:
            pf += int(info.ri_phys_footprint)
            res += int(info.ri_resident_size)
            pagein += int(info.ri_pageins)
            diskread += int(info.ri_diskio_bytesread)
        else:
            try:
                pf += psutil.Process(pid).memory_info().rss
                method = "rss"
            except psutil.Error:
                pass
    return {
        "phys_footprint_b": pf,
        "resident_b": res,
        "pageins_b": pagein,
        "diskio_bytesread_b": diskread,
        "mem_method": method,
    }


def swap_used_gb() -> Optional[float]:
    """Parse `vm.swapusage` -> used swap in GB. Format: 'total = 8192.00M  used = 7310.75M ...'"""
    try:
        out = subprocess.check_output(["sysctl", "-n", "vm.swapusage"]).decode()
        parts = out.replace("=", " ").split()  # [..., 'used', '7310.75M', ...]
        for i, t in enumerate(parts):
            if t == "used" and i + 1 < len(parts):
                v = parts[i + 1]
                mult = {"M": 1.0, "G": 1024.0, "K": 1.0 / 1024.0}.get(v[-1])
                if mult is None:
                    return None
                return float(v[:-1]) * mult / 1024.0  # MB -> GB
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# 8 GB fixed ballast (random fill + periodic re-touch so the compressor can't purge it)
# ---------------------------------------------------------------------------


class Ballast:
    """Exactly BALLAST_CPU_GB + BALLAST_MLX_GB of resident competing working set.

    Plan-review R1: this is a FIXED allocation, NOT create_memory_pressure (which leaves N
    GB *available*). Plan-review R4: CPU half is random-filled (non-compressible) and the
    `retouch()` method must be called periodically so the macOS memory compressor cannot
    silently compress/purge the pages and shrink the effective working set.
    """

    def __init__(self):
        self.cpu = None
        self.mlx = None
        self._page = os.sysconf("SC_PAGE_SIZE")
        self._n = 0

    def allocate(self) -> dict:
        before = get_available_memory_gb()
        # CPU ballast: allocate then overwrite with a non-compressible random pattern.
        self.cpu = allocate_cpu_ballast(BALLAST_CPU_GB)  # touched 0xFF per page
        rnd = os.urandom(self._page)
        for off in range(0, len(self.cpu), self._page):
            self.cpu[off : off + len(rnd)] = rnd[: max(0, min(len(rnd), len(self.cpu) - off))]
        self.mlx = allocate_mlx_ballast(BALLAST_MLX_GB)  # Metal, force-eval'd
        after = get_available_memory_gb()
        return {
            "ballast_cpu_gb": BALLAST_CPU_GB,
            "ballast_mlx_gb": BALLAST_MLX_GB,
            "ballast_total_gb": BALLAST_CPU_GB + BALLAST_MLX_GB,
            "avail_before_gb": round(before, 2),
            "avail_after_gb": round(after, 2),
            "avail_drop_gb": round(before - after, 2),
        }

    def retouch(self):
        """Read+write one byte per page so the pages stay hot/resident."""
        if self.cpu is None:
            return
        b = self.cpu
        for off in range(0, len(b), self._page):
            b[off] = (b[off] + 1) & 0xFF
        self._n += 1


# ---------------------------------------------------------------------------
# llama-server lifecycle
# ---------------------------------------------------------------------------


def start_server(ngl: int, log_path: Path) -> subprocess.Popen:
    args = [
        LLAMA_SERVER,
        "-m", MODEL_PATH,
        "--port", str(PORT),
        "--ctx-size", str(CTX_SIZE),
        "--batch-size", str(BATCH),
        "--ubatch-size", str(UBATCH),
        "--parallel", str(PARALLEL),
        "-ngl", str(ngl),
        "--jinja",
        "--metrics",
        # mmap ENABLED (default): do NOT pass --no-mmap or --mlock.
    ]
    f = open(log_path, "w")
    proc = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT)
    return proc


def wait_healthy(base_url: str, proc: subprocess.Popen, timeout_s: float) -> bool:
    import requests

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        if proc.poll() is not None:
            return False  # server died during load
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def parse_buffer_sizes(log_path: Path) -> dict:
    """Pull CPU/Metal model buffer + KV cache sizes from the llama.cpp load banner."""
    out = {"metal_model_mib": None, "cpu_model_mib": None, "kv_mib": None, "raw": []}
    try:
        text = log_path.read_text(errors="ignore")
    except OSError:
        return out
    for line in text.splitlines():
        low = line.lower()
        if "buffer size" in low or "kv self size" in low or "model buffer" in low:
            out["raw"].append(line.strip())
    return out


# ---------------------------------------------------------------------------
# Calibration: verify ri_diskio_bytesread actually captures mmap cold faults
# ---------------------------------------------------------------------------


def calibrate() -> dict:
    """Plan-review R6: confirm ri_diskio_bytesread rises on cold mmap faults, ~0 on warm.

    mmap-scan the GGUF itself (cold via purge if available, then warm) and compare the
    per-process diskio delta. If cold/warm < 2x, the counter is informational-only and the
    pageout gate becomes the primary churn signal.
    """
    import mmap

    path = MODEL_PATH if os.path.exists(MODEL_PATH) else __file__
    size = os.path.getsize(path)
    scan = min(size, 1 * 1024**3)  # scan up to 1 GB

    def mmap_scan() -> int:
        before = tree_counters(os.getpid())["diskio_bytesread_b"]
        with open(path, "rb") as fh:
            mm = mmap.mmap(fh.fileno(), scan, prot=mmap.PROT_READ)
            acc = 0
            for off in range(0, scan, 1024 * 1024):
                acc += mm[off]
            mm.close()
        after = tree_counters(os.getpid())["diskio_bytesread_b"]
        return after - before

    # Best-effort cold: purge the page cache (needs sudo; ignore failure).
    subprocess.run(["purge"], capture_output=True)
    cold = mmap_scan()
    warm = mmap_scan()
    ratio = (cold / warm) if warm > 0 else (float("inf") if cold > 0 else 0.0)
    usable = ratio >= 2.0 or (cold > 50 * 1024**2 and warm < cold / 2)
    result = {
        "scan_bytes": scan,
        "cold_diskread_mb": round(cold / 1024**2, 2),
        "warm_diskread_mb": round(warm / 1024**2, 2),
        "cold_warm_ratio": round(ratio, 2) if ratio != float("inf") else "inf",
        "ri_diskio_usable_as_churn_signal": bool(usable),
        "note": (
            "ri_diskio_bytesread captures mmap cold faults; usable as primary churn gate."
            if usable
            else "ri_diskio_bytesread did NOT separate cold/warm >=2x; treat as informational, "
            "use pageout gate as primary churn signal."
        ),
    }
    print(f"[calibrate] cold={result['cold_diskread_mb']}MB warm={result['warm_diskread_mb']}MB "
          f"ratio={result['cold_warm_ratio']} usable={usable}")
    log_experiment(
        experiment_name="h9_e1b_diskio_calibration",
        phase="E1b-A",
        config={"scan_bytes": scan, "path": os.path.basename(path)},
        results=result,
        status="completed",
    )
    return result


# ---------------------------------------------------------------------------
# Sustained co-residency run
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    t: float
    phys_gb: float
    resident_gb: float
    pageins_b: int
    diskread_b: int
    swap_gb: Optional[float]
    avail_gb: float


@dataclass
class DecodeResult:
    ttft_ms: Optional[float]
    decode_tok_s: Optional[float]


def run(ngl: int, order: str, minutes: float) -> dict:
    import requests

    base_url = f"http://127.0.0.1:{PORT}"
    server_log = LOGDIR / f"server_ngl{ngl}_{order}.log"
    LOGDIR.mkdir(parents=True, exist_ok=True)

    ballast = Ballast()
    ballast_info: dict = {}
    vm_before = get_vm_stat()

    # --- Load order (plan-review R4) -------------------------------------------------
    if order == "B2":  # ballast-then-load (load-under-pressure)
        print("[run] B2: allocating 8 GB ballast BEFORE model load")
        ballast_info = ballast.allocate()

    print(f"[run] starting llama-server -ngl {ngl} (mmap on, ctx {CTX_SIZE})")
    proc = start_server(ngl, server_log)
    load_t0 = time.perf_counter()
    healthy = wait_healthy(base_url, proc, timeout_s=900)
    load_s = time.perf_counter() - load_t0

    if not healthy:
        status = "load_failed"
        print(f"[run] LOAD FAILED after {load_s:.0f}s (server exited or never healthy)")
        _kill(proc)
        result = {
            "verdict": "load_failed",
            "order": order,
            "ngl": ngl,
            "load_seconds": round(load_s, 1),
            "ballast": ballast_info,
            "buffer_sizes": parse_buffer_sizes(server_log),
        }
        _log(ngl, order, minutes, result, status="load_failed")
        return result

    print(f"[run] server healthy after {load_s:.0f}s")
    buffers = parse_buffer_sizes(server_log)

    if order == "B1":  # load-then-ballast (post-load survival)
        print("[run] B1: allocating 8 GB ballast AFTER model load")
        ballast_info = ballast.allocate()

    # Standardized warmup (plan-review R2): one full decode so all configs start warm.
    try:
        served_generate_timed(
            base_url, [{"role": "user", "content": "Say hello."}], max_tokens=16
        )
    except Exception as e:
        print(f"[run] warmup request errored: {e}")

    # --- Sustained generation + sampling loop ---------------------------------------
    samples: list[Sample] = []
    decodes: list[DecodeResult] = []
    root = proc.pid
    end_t = time.perf_counter() + minutes * 60
    next_sample = time.perf_counter()
    print(f"[run] sustained loop for {minutes} min, sampling every {SAMPLE_INTERVAL_S}s")

    while time.perf_counter() < end_t:
        # One back-to-back decode request (keeps the model continuously generating).
        try:
            timed = served_generate_timed(
                base_url,
                [{"role": "user", "content": DECODE_PROMPT}],
                max_tokens=DECODE_MAX_TOKENS,
            )
            decodes.append(DecodeResult(timed.get("ttft_ms"), timed.get("decode_tok_s")))
        except Exception as e:
            print(f"[run] decode request errored: {e}")
            decodes.append(DecodeResult(None, None))

        now = time.perf_counter()
        if now >= next_sample:
            c = tree_counters(root)
            samples.append(
                Sample(
                    t=now,
                    phys_gb=c["phys_footprint_b"] / 1024**3,
                    resident_gb=c["resident_b"] / 1024**3,
                    pageins_b=c["pageins_b"],
                    diskread_b=c["diskio_bytesread_b"],
                    swap_gb=swap_used_gb(),
                    avail_gb=get_available_memory_gb(),
                )
            )
            ballast.retouch()  # keep ballast resident
            s = samples[-1]
            print(f"  [{int(now - (end_t - minutes*60))}s] phys={s.phys_gb:.2f}GB "
                  f"avail={s.avail_gb:.2f}GB swap={s.swap_gb}GB diskread="
                  f"{s.diskread_b/1024**3:.2f}GB")
            next_sample = now + SAMPLE_INTERVAL_S

    elapsed = samples[-1].t - samples[0].t if len(samples) >= 2 else minutes * 60
    vm_after = get_vm_stat()
    vmd = vm_stat_delta(vm_before, vm_after)

    result = _compute_gates(ngl, order, minutes, samples, decodes, elapsed, vmd, ballast_info, buffers, load_s)
    _kill(proc)
    _log(ngl, order, minutes, result)
    return result


def _compute_gates(ngl, order, minutes, samples, decodes, elapsed, vmd, ballast_info, buffers, load_s) -> dict:
    phys = [s.phys_gb for s in samples] or [0.0]
    phys_sorted = sorted(phys)
    p95 = phys_sorted[min(len(phys_sorted) - 1, int(0.95 * (len(phys_sorted) - 1)))]
    phys_max = max(phys)
    phys_med = statistics.median(phys)

    tps = [d.decode_tok_s for d in decodes if d.decode_tok_s]
    ttfts = [d.ttft_ms for d in decodes if d.ttft_ms]
    decode_med = statistics.median(tps) if tps else None
    ttft_p50_s = (statistics.median(ttfts) / 1000.0) if ttfts else None

    # Disk-read churn over the back-half (exclude load/warmup).
    half = samples[len(samples) // 2 :] if len(samples) >= 4 else samples
    churn_mb_s = None
    if len(half) >= 2:
        dt = half[-1].t - half[0].t
        db = half[-1].diskread_b - half[0].diskread_b
        churn_mb_s = (db / 1024**2) / dt if dt > 0 else None

    hours = elapsed / 3600.0 if elapsed > 0 else None
    pageout_mb_hr = (vmd["pageout_delta_mb"] / hours) if hours else None

    g_decode = decode_med is not None and decode_med >= 12
    g_ttft = ttft_p50_s is not None and ttft_p50_s <= 10
    g_pageout = pageout_mb_hr is not None and pageout_mb_hr < 200
    g_phys = phys_max <= 8.0 and p95 <= 8.0
    g_churn = churn_mb_s is not None and churn_mb_s < 50
    # tool-call gate is run separately via h9_e1_harness.py; recorded as None here.

    gates = {
        "decode_tok_s>=12": g_decode,
        "ttft_p50_s<=10": g_ttft,
        "pageout_mb_hr<200": g_pageout,
        "phys_p95_and_max<=8": g_phys,
        "diskread_mb_s<50": g_churn,
        "tool_call>=90pct": None,  # filled by harness
    }
    perf_pass = all(v for v in [g_decode, g_ttft, g_pageout, g_phys, g_churn])

    return {
        "verdict_perf_gates": "pass" if perf_pass else "fail",
        "note": "tool-call gate run separately via h9_e1_harness.py --runtime served",
        "order": order,
        "ngl": ngl,
        "minutes": minutes,
        "load_seconds": round(load_s, 1),
        "samples": len(samples),
        "decode_tok_s_median": round(decode_med, 2) if decode_med else None,
        "ttft_p50_s": round(ttft_p50_s, 2) if ttft_p50_s else None,
        "phys_footprint_gb": {
            "median": round(phys_med, 2),
            "p95": round(p95, 2),
            "max": round(phys_max, 2),
        },
        "diskread_churn_mb_s": round(churn_mb_s, 2) if churn_mb_s is not None else None,
        "pageout_mb_hr": round(pageout_mb_hr, 1) if pageout_mb_hr is not None else None,
        "pagein_mb_total": vmd["pagein_delta_mb"],  # diagnostic, NOT a gate
        "swap_used_gb_last": samples[-1].swap_gb if samples else None,
        "avail_gb_last": round(samples[-1].avail_gb, 2) if samples else None,
        "gates": gates,
        "ballast": ballast_info,
        "buffer_sizes": buffers,
    }


def _kill(proc: subprocess.Popen):
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def _log(ngl, order, minutes, result, status="completed"):
    log_experiment(
        experiment_name=f"h9_e1b_coresidency_ngl{ngl}_{order}",
        phase="E1b-B",
        config={
            "model_repo": MODEL_REPO,
            "model_file": MODEL_FILE,
            "model_sha256": MODEL_SHA256,
            "quant": "Q4_K_M",
            "runtime": "llama.cpp",
            "mmap": True,
            "mlock": False,
            "ngl": ngl,
            "ctx_size": CTX_SIZE,
            "batch": BATCH,
            "ubatch": UBATCH,
            "parallel": PARALLEL,
            "kv_cache": "f16",
            "load_order": order,
            "minutes": minutes,
            "ballast_total_gb": BALLAST_CPU_GB + BALLAST_MLX_GB,
            "decode_prompt_tokens_approx": 512,
            "decode_max_tokens": DECODE_MAX_TOKENS,
            "sample_interval_s": SAMPLE_INTERVAL_S,
            # project-convention fields (value-or-null+reason):
            "cache_hit_rate": None,  # reason: no GGUF page-residency API; see diskread/pageins
            "perplexity": None,  # reason: quality gated via 20-case tool-call harness
        },
        results={**result, "mem_method": "phys_footprint"},
        status=status,
    )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("calibrate")
    r = sub.add_parser("run")
    r.add_argument("--ngl", type=int, required=True)
    r.add_argument("--order", choices=["B1", "B2"], default="B1")
    r.add_argument("--minutes", type=float, default=30.0)
    args = ap.parse_args()

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: GGUF not found at {MODEL_PATH}. Download first.", file=sys.stderr)
        sys.exit(2)

    if args.cmd == "calibrate":
        calibrate()
    elif args.cmd == "run":
        res = run(args.ngl, args.order, args.minutes)
        print("\n=== RESULT ===")
        import json as _j
        print(_j.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
