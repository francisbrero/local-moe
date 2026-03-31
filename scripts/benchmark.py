"""
benchmark.py — Agent-editable configuration knobs.

The AI agent modifies ONLY this file between experiments.
Each knob below has a live code path in run.py.

To run an experiment: edit knobs here, then `uv run python scripts/run.py`
"""

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------

# Model tier: S (0.5B), M (3B), L (7B), XL (14B)
MODEL_TIER = "S"

# Override the tier's default repo (None = use MODEL_TIERS[MODEL_TIER])
MODEL_REPO = None

# Pin a specific HF commit SHA (None = latest cached revision)
MODEL_REVISION = None

# ---------------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------------

# Maximum tokens to generate per inference call
MAX_TOKENS = 256

# Context length for KV cache estimation
CONTEXT_LENGTH = 2048

# Number of warm repetitions (steady-state measurement)
REPETITIONS = 3

# ---------------------------------------------------------------------------
# Optimization knobs
# ---------------------------------------------------------------------------

# Enable mx.compile() on the model
COMPILE_MODEL = False

# ---------------------------------------------------------------------------
# Experiment control
# ---------------------------------------------------------------------------

# Wall-clock time budget per experiment (seconds). Abort if exceeded.
TIME_BUDGET_SECONDS = 300

# Experiment name (must be unique per run)
EXPERIMENT_NAME = "baseline_S"

# What are we testing?
HYPOTHESIS = "Baseline measurement for S-tier model"
