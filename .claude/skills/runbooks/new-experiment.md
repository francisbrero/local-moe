---
description: Runbook for running automated inference experiments
globs: []
alwaysApply: false
---

# New Experiment Runbook

## Goal

Maximize `warm.tok_s` while constraining `quality.perplexity_mean` within **5% of the same model_repo + model_revision baseline**.

## Architecture

```
scripts/
├── prepare.py    # Immutable: hw validation, model registry, metric utils, prompts
├── benchmark.py  # Mutable: agent-editable config knobs (module-level constants)
└── run.py        # Executor: loads model, runs inference, collects metrics, logs JSONL
```

- **prepare.py** — DO NOT EDIT. Hardware validation, model tiers, memory estimation, metric helpers.
- **benchmark.py** — EDIT THIS FILE. All experiment knobs are module-level constants.
- **run.py** — DO NOT EDIT. Reads knobs from benchmark.py, runs cold/warm/quality phases, logs to `experiments.jsonl`.

## Live Knobs (benchmark.py)

| Knob | Type | Default | Description |
|------|------|---------|-------------|
| `MODEL_TIER` | str | `"S"` | S (0.5B), M (3B), L (7B), XL (14B) |
| `MODEL_REPO` | str\|None | `None` | Override tier's default HF repo |
| `MODEL_REVISION` | str\|None | `None` | Pin specific HF commit SHA |
| `MAX_TOKENS` | int | `256` | Max tokens per generation |
| `CONTEXT_LENGTH` | int | `2048` | Context length for KV cache estimate |
| `REPETITIONS` | int | `3` | Number of warm measurement reps |
| `COMPILE_MODEL` | bool | `False` | Enable mx.compile() |
| `TIME_BUDGET_SECONDS` | int | `300` | Wall-clock abort limit |
| `EXPERIMENT_NAME` | str | — | Unique name for this run |
| `HYPOTHESIS` | str | — | What you're testing |

**NOT a knob**: Perplexity always runs. The quality gate cannot be disabled.

## Rules

1. **One knob at a time** — change a single knob per experiment to isolate effects
2. **Unique names** — every experiment must have a unique `EXPERIMENT_NAME`
3. **Baseline first** — run a baseline for each model identity (repo + revision) before tuning
4. **Pin revision on baseline** — after the first baseline, set `MODEL_REVISION` to the resolved SHA from the record's `config.model_revision` to ensure reproducibility
5. **Reject failures** — all-truncated warm runs and aborted experiments don't count as valid
6. **Log everything** — even failed experiments are valuable data

## Strategy

1. **Start with S tier** — fast iteration, cheap experiments
2. **Establish baseline** — run with defaults, record the revision SHA
3. **Tune generation** — try `COMPILE_MODEL=True`, adjust `MAX_TOKENS`, etc.
4. **Scale up** — move to M, L, XL once S is optimized
5. **Compare** — always measure against the same-revision baseline

## Workflow

```bash
# 1. Download model (only time network is used)
uv run python scripts/prepare.py --tier S

# 2. Edit benchmark.py knobs
# (change EXPERIMENT_NAME, HYPOTHESIS, and one knob)

# 3. Run experiment
uv run python scripts/run.py

# 4. Check results
tail -1 experiments.jsonl | python3 -m json.tool

# 5. Compare against baseline
python3 -c "
import json
for line in open('experiments.jsonl'):
    r = json.loads(line)
    if r['status'] == 'completed':
        print(f\"{r['meta']['experiment']}: {r['warm']['tok_s']} tok/s, ppl={r['quality']['perplexity_mean']}\")
"
```

## Record Schema

Completed experiments contain: `status`, `cold`, `warm`, `peak`, `quality`, `config`, `env`, `meta`.

Aborted experiments contain: `status`, `abort_reason`, `config`, `env`, `meta`, plus any partial data collected before abort.

## Experiment Documentation

For longer experiments, use the `dev/active/` workflow:

```bash
mkdir -p dev/active/<experiment-name>
cp dev/templates/plan.md dev/active/<experiment-name>/plan.md
cp dev/templates/context.md dev/active/<experiment-name>/context.md
```

Archive when done: `mv dev/active/<name> dev/completed/`
