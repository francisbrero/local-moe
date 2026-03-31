"""
prepare.py — Immutable infrastructure for the experiment harness.

Hardware validation, model registry, metric utilities, prompts.
This file is NOT edited by the agent — only benchmark.py has live knobs.
"""

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import psutil

# ---------------------------------------------------------------------------
# Hardware validation
# ---------------------------------------------------------------------------


def validate_hardware():
    """Assert arm64 Apple Silicon, report chip + memory."""
    machine = platform.machine()
    if machine != "arm64":
        sys.exit(f"Unsupported architecture: {machine}. Requires arm64 (Apple Silicon).")

    info = get_environment_info()
    print(f"Hardware OK: {info['chip']}, {info['memory_gb']:.1f} GB unified memory")
    print(f"macOS {info['macos_version']}, Python {platform.python_version()}")
    return info


# ---------------------------------------------------------------------------
# Environment info
# ---------------------------------------------------------------------------


def get_environment_info() -> dict:
    """Gather git SHA, mlx/mlx-lm/macOS versions, chip, memory_gb."""
    import mlx.core as mx

    try:
        import mlx_lm

        mlx_lm_version = mlx_lm.__version__
    except AttributeError:
        mlx_lm_version = "unknown"

    # Git SHA
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

    # Chip name
    try:
        chip = (
            subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        chip = platform.processor() or "unknown"

    mem_bytes = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    memory_gb = mem_bytes / (1024**3)

    return {
        "git_sha": git_sha,
        "mlx_version": mx.__version__,
        "mlx_lm_version": mlx_lm_version,
        "macos_version": platform.mac_ver()[0],
        "chip": chip,
        "memory_gb": round(memory_gb, 2),
    }


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_TIERS = {
    "S": {
        "hf_repo": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        "approx_size_gb": 0.4,
        "type": "4bit",
        "params": "0.5B",
    },
    "M": {
        "hf_repo": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "approx_size_gb": 1.8,
        "type": "4bit",
        "params": "3B",
    },
    "L": {
        "hf_repo": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "approx_size_gb": 4.5,
        "type": "4bit",
        "params": "7B",
    },
    "XL": {
        "hf_repo": "mlx-community/Qwen2.5-14B-Instruct-4bit",
        "approx_size_gb": 8.5,
        "type": "4bit",
        "params": "14B",
    },
}


# ---------------------------------------------------------------------------
# HF cache helpers (offline-only)
# ---------------------------------------------------------------------------

_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def _repo_dir(repo: str) -> Path:
    """Return the HF cache directory for a repo."""
    # HF cache uses models--<org>--<model> format
    return _HF_CACHE / f"models--{repo.replace('/', '--')}"


def is_model_cached(repo: str, revision: str | None = None) -> bool:
    """Check if a model repo (and optionally exact revision) is in HF cache."""
    repo_path = _repo_dir(repo)
    if not repo_path.exists():
        return False
    if revision is None:
        # Any snapshot counts
        snapshots = repo_path / "snapshots"
        return snapshots.exists() and any(snapshots.iterdir())
    # Check for exact revision
    snapshot = repo_path / "snapshots" / revision
    return snapshot.exists()


def get_model_revision(repo: str, revision: str | None = None) -> str:
    """Resolve HF commit SHA from local cache. Raises if not cached."""
    repo_path = _repo_dir(repo)
    snapshots = repo_path / "snapshots"
    if not snapshots.exists():
        raise FileNotFoundError(f"Model not cached: {repo}")

    if revision is not None:
        snapshot = snapshots / revision
        if not snapshot.exists():
            raise FileNotFoundError(
                f"Revision {revision} not cached for {repo}. "
                f"Available: {[d.name for d in snapshots.iterdir() if d.is_dir()]}"
            )
        return revision

    # Return the most recently modified snapshot
    dirs = sorted(
        [d for d in snapshots.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not dirs:
        raise FileNotFoundError(f"No snapshots cached for {repo}")
    return dirs[0].name


def get_model_config(repo: str, revision: str | None = None) -> dict:
    """Read config.json from local HF cache only. No network.

    Returns dict with: n_layers, hidden_size, n_kv_heads, head_dim.
    Raises FileNotFoundError if not cached.
    """
    rev = get_model_revision(repo, revision)
    config_path = _repo_dir(repo) / "snapshots" / rev / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found at {config_path}")

    with open(config_path) as f:
        raw = json.load(f)

    # Normalize across model architectures
    n_layers = raw.get("num_hidden_layers", raw.get("n_layer", 0))
    hidden_size = raw.get("hidden_size", raw.get("d_model", 0))
    n_heads = raw.get("num_attention_heads", raw.get("n_head", 0))
    n_kv_heads = raw.get("num_key_value_heads", n_heads)
    head_dim = raw.get("head_dim", hidden_size // n_heads if n_heads else 0)

    return {
        "n_layers": n_layers,
        "hidden_size": hidden_size,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
    }


# ---------------------------------------------------------------------------
# Model download (the ONLY function that touches the network)
# ---------------------------------------------------------------------------


def download_model(tier: str, revision: str | None = None):
    """Download a model via mlx_lm.load() to populate HF cache.

    This is the ONLY function in the harness that performs network I/O.
    """
    import mlx_lm

    if tier not in MODEL_TIERS:
        sys.exit(f"Unknown tier: {tier}. Choose from {list(MODEL_TIERS.keys())}")

    info = MODEL_TIERS[tier]
    repo = info["hf_repo"]

    if is_model_cached(repo, revision):
        print(f"Model already cached: {repo}" + (f" @ {revision}" if revision else ""))
        return

    print(f"Downloading {tier} ({info['params']}, ~{info['approx_size_gb']}GB): {repo}")
    kwargs = {}
    if revision:
        kwargs["revision"] = revision
    mlx_lm.load(repo, **kwargs)
    print(f"Download complete: {repo}")


# ---------------------------------------------------------------------------
# Memory estimation & safety
# ---------------------------------------------------------------------------


def estimate_memory_gb(
    approx_size_gb: float, model_config: dict, context_length: int
) -> float:
    """Estimate total memory: model weights + KV cache + overhead.

    KV cache = n_layers * 2 * n_kv_heads * head_dim * context_length * 2 / 1e9
    (K + V, float16 = 2 bytes per element)
    """
    n_layers = model_config["n_layers"]
    n_kv_heads = model_config["n_kv_heads"]
    head_dim = model_config["head_dim"]

    kv_bytes = n_layers * 2 * n_kv_heads * head_dim * context_length * 2
    kv_gb = kv_bytes / 1e9

    overhead_gb = 1.0
    total = approx_size_gb + kv_gb + overhead_gb
    return round(total, 2)


def check_memory_budget(estimate_gb: float) -> tuple[bool, str]:
    """Check estimated memory against hw.memsize - 5GB headroom.

    Returns (ok, message).
    """
    mem_bytes = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    total_gb = mem_bytes / (1024**3)
    budget_gb = total_gb - 5.0

    if estimate_gb > budget_gb:
        return False, (
            f"Estimated {estimate_gb:.1f}GB exceeds budget {budget_gb:.1f}GB "
            f"(total {total_gb:.1f}GB - 5GB headroom)"
        )
    return True, f"OK: {estimate_gb:.1f}GB within {budget_gb:.1f}GB budget"


def is_memory_safe() -> bool:
    """Return True iff psutil.virtual_memory().available >= 2GB.

    On macOS unified memory, `available` reflects pressure from both
    RSS and Metal/GPU allocations, so this is the single authoritative signal.
    """
    return psutil.virtual_memory().available >= 2 * 1024**3


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def get_peak_rss_mb() -> float:
    """Current peak RSS in MB (self process)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS returns bytes, Linux returns KB
    if platform.system() == "Darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


def get_disk_io_snapshot() -> dict:
    """Snapshot of disk I/O counters."""
    counters = psutil.disk_io_counters()
    if counters is None:
        return {"read_bytes": 0, "write_bytes": 0}
    return {"read_bytes": counters.read_bytes, "write_bytes": counters.write_bytes}


def compute_ssd_read_gb(before: dict, after: dict) -> float:
    """Compute SSD read between two snapshots in GB."""
    delta = after["read_bytes"] - before["read_bytes"]
    return round(delta / (1024**3), 3)


def get_peak_gpu_mb() -> float | None:
    """Get current active GPU/Metal memory in MB. Returns None if unavailable.

    On macOS unified memory, GPU and CPU share the same pool.
    """
    try:
        import mlx.core as mx

        info = mx.metal.get_active_memory()
        return round(info / (1024 * 1024), 1)
    except (ImportError, AttributeError):
        return None


def reset_peak_memory():
    """Reset MLX peak memory counter if available."""
    try:
        import mlx.core as mx

        # Use new API (mx.reset_peak_memory), fall back to deprecated
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        else:
            mx.metal.reset_peak_memory()
    except (ImportError, AttributeError):
        pass


def get_peak_gpu_mb_peak() -> float | None:
    """Get peak GPU memory since last reset in MB."""
    try:
        import mlx.core as mx

        # Use new API (mx.get_peak_memory), fall back to deprecated
        if hasattr(mx, "get_peak_memory"):
            return round(mx.get_peak_memory() / (1024 * 1024), 1)
        return round(mx.metal.get_peak_memory() / (1024 * 1024), 1)
    except (ImportError, AttributeError):
        return None


def compute_perplexity(model, tokenizer, text: str) -> float:
    """Compute perplexity of text using the model.

    Uses a sliding window approach with the model's context length.
    """
    import mlx.core as mx
    import mlx.nn as nn

    tokens = tokenizer.encode(text)
    if len(tokens) < 2:
        return float("inf")

    tokens_array = mx.array([tokens])
    logits = model(tokens_array)

    # Shift: predict token[i+1] from position[i]
    shift_logits = logits[:, :-1, :]
    shift_labels = tokens_array[:, 1:]

    loss = nn.losses.cross_entropy(shift_logits, shift_labels, reduction="mean")
    mx.eval(loss)
    return round(float(mx.exp(loss).item()), 4)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

BENCH_PROMPT = (
    "Explain the key principles of distributed computing in simple terms. "
    "Cover consistency, availability, partition tolerance, and how they relate "
    "to real-world system design."
)

PERPLEXITY_PASSAGES = {
    "prose": (
        "The quick brown fox jumps over the lazy dog near the riverbank. "
        "As the sun set behind the mountains, the village came alive with the "
        "sounds of evening — crickets chirping, children laughing, and the distant "
        "hum of a tractor returning from the fields. The old oak tree stood watch "
        "over everything, its branches heavy with the weight of centuries."
    ),
    "code": (
        "def fibonacci(n):\n"
        '    """Return the nth Fibonacci number using dynamic programming."""\n'
        "    if n <= 1:\n"
        "        return n\n"
        "    dp = [0] * (n + 1)\n"
        "    dp[1] = 1\n"
        "    for i in range(2, n + 1):\n"
        "        dp[i] = dp[i-1] + dp[i-2]\n"
        "    return dp[n]\n"
        "\n"
        "# Test the function\n"
        "for i in range(10):\n"
        '    print(f"F({i}) = {fibonacci(i)}")\n'
    ),
    "qa": (
        "Question: What is the capital of France?\n"
        "Answer: The capital of France is Paris. It is located in the north-central "
        "part of the country along the Seine River. Paris is the largest city in France "
        "with a population of over 2 million in the city proper and over 12 million "
        "in the metropolitan area."
    ),
    "reasoning": (
        "If all cats are mammals, and all mammals are animals, then all cats are animals. "
        "This is an example of a syllogism, a form of deductive reasoning. "
        "Given that Socrates is a man, and all men are mortal, we can conclude that "
        "Socrates is mortal. The logical structure is: All A are B, All B are C, "
        "therefore All A are C."
    ),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Prepare models and validate hardware")
    parser.add_argument(
        "--tier",
        choices=list(MODEL_TIERS.keys()),
        default="S",
        help="Model tier to download (default: S)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_tiers",
        help="Download all model tiers",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Pin a specific HF commit SHA",
    )
    args = parser.parse_args()

    # Always validate hardware first
    info = validate_hardware()
    print(f"MLX {info['mlx_version']}, mlx-lm {info['mlx_lm_version']}")
    print()

    if args.all_tiers:
        for tier in MODEL_TIERS:
            t = MODEL_TIERS[tier]
            ok, msg = check_memory_budget(t["approx_size_gb"] + 1.0)
            if not ok:
                print(f"Skipping {tier} ({t['params']}): {msg}")
                continue
            download_model(tier, revision=args.revision)
            print()
    else:
        download_model(args.tier, revision=args.revision)

    print("Done. Models cached in ~/.cache/huggingface/hub/")


if __name__ == "__main__":
    main()
