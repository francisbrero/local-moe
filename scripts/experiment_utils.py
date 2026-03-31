"""
experiment_utils.py — Shared utilities for expert offloading experiment scripts.

Provides structured logging, memory pressure simulation, and metric collection
used across Phases 0-4a.
"""

import ctypes
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

EXPERIMENTS_FILE = Path(__file__).parent.parent / "experiments.jsonl"


# ---------------------------------------------------------------------------
# Environment info
# ---------------------------------------------------------------------------


def get_environment_info() -> dict:
    """Gather hardware info: chip, memory, OS version, git SHA."""
    try:
        chip = (
            subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        chip = platform.processor() or "unknown"

    try:
        git_sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_sha = "unknown"

    mem_bytes = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    memory_gb = mem_bytes / (1024**3)

    return {
        "chip": chip,
        "memory_gb": round(memory_gb, 2),
        "macos_version": platform.mac_ver()[0],
        "python_version": platform.python_version(),
        "git_sha": git_sha,
        "available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
    }


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------


def log_experiment(
    experiment_name: str,
    phase: str,
    config: dict,
    results: dict,
    status: str = "completed",
    env: dict | None = None,
):
    """Append a structured JSON record to experiments.jsonl."""
    if env is None:
        env = get_environment_info()

    record = {
        "experiment_name": experiment_name,
        "phase": phase,
        "status": status,
        "config": config,
        "results": results,
        "env": env,
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    with open(EXPERIMENTS_FILE, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"  Logged: {experiment_name} -> {EXPERIMENTS_FILE}")
    return record


# ---------------------------------------------------------------------------
# Memory pressure simulation
# ---------------------------------------------------------------------------


def get_available_memory_gb() -> float:
    """Return available memory in GB via psutil."""
    return psutil.virtual_memory().available / (1024**3)


def get_vm_stat() -> dict:
    """Parse vm_stat output for pageout/pagein counters."""
    try:
        output = subprocess.check_output(["vm_stat"]).decode()
        stats = {}
        for line in output.strip().split("\n"):
            if ":" in line:
                key, _, val = line.partition(":")
                val = val.strip().rstrip(".")
                try:
                    stats[key.strip()] = int(val)
                except ValueError:
                    pass
        # Convert to bytes (macOS page size = 16384 on arm64)
        page_size = os.sysconf("SC_PAGE_SIZE")
        return {
            "pageins": stats.get("Pageins", 0) * page_size,
            "pageouts": stats.get("Pageouts", 0) * page_size,
            "page_size": page_size,
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"pageins": 0, "pageouts": 0, "page_size": 16384}


def vm_stat_delta(before: dict, after: dict) -> dict:
    """Compute pageout/pagein deltas in MB."""
    return {
        "pagein_delta_mb": round(
            (after["pageins"] - before["pageins"]) / (1024 * 1024), 2
        ),
        "pageout_delta_mb": round(
            (after["pageouts"] - before["pageouts"]) / (1024 * 1024), 2
        ),
    }


def allocate_cpu_ballast(size_gb: float) -> memoryview:
    """Allocate and force-touch CPU memory ballast.

    Returns a memoryview over force-touched memory. The caller must keep
    a reference to prevent garbage collection.
    """
    size_bytes = int(size_gb * 1024**3)
    buf = bytearray(size_bytes)
    # Force-touch every page to ensure pages are faulted in
    page_size = os.sysconf("SC_PAGE_SIZE")
    mv = memoryview(buf)
    for offset in range(0, size_bytes, page_size):
        buf[offset] = 0xFF
    return mv


def allocate_mlx_ballast(size_gb: float):
    """Allocate and force-evaluate MLX Metal buffer.

    Returns the MLX array. Caller must keep a reference.
    """
    import mlx.core as mx

    # Allocate as float32 (4 bytes each)
    n_elements = int(size_gb * 1024**3 / 4)
    arr = mx.ones(n_elements, dtype=mx.float32)
    mx.eval(arr)  # Force materialization on Metal
    return arr


def create_memory_pressure(target_available_gb: float) -> tuple:
    """Create memory pressure to reach target available memory.

    Returns (cpu_ballast, mlx_ballast) — caller must keep references.
    Returns (None, None) if already at or below target.
    """
    current = get_available_memory_gb()
    if current <= target_available_gb:
        print(f"  Already at {current:.1f}GB available (target: {target_available_gb}GB)")
        return None, None

    to_allocate = current - target_available_gb
    # Split 50/50 between CPU and Metal to simulate real inference
    cpu_gb = to_allocate / 2
    mlx_gb = to_allocate / 2

    print(f"  Allocating pressure: {cpu_gb:.1f}GB CPU + {mlx_gb:.1f}GB Metal")
    cpu_ballast = allocate_cpu_ballast(cpu_gb)
    mlx_ballast = allocate_mlx_ballast(mlx_gb)

    actual = get_available_memory_gb()
    print(f"  Available after pressure: {actual:.1f}GB (target: {target_available_gb}GB)")
    return cpu_ballast, mlx_ballast


# ---------------------------------------------------------------------------
# Peak RSS helper
# ---------------------------------------------------------------------------


def get_peak_rss_mb() -> float:
    """Current peak RSS in MB (self process)."""
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)
    if platform.system() == "Darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


def get_rss_mb() -> float:
    """Current RSS in MB (not peak)."""
    return psutil.Process().memory_info().rss / (1024 * 1024)
