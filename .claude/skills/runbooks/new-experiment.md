---
description: Runbook for setting up and running a new inference experiment
globs: []
alwaysApply: false
---

# New Experiment Runbook

## 1. Setup

```bash
# Create experiment directory
mkdir -p dev/active/<experiment-name>

# Create docs from templates
cp dev/templates/plan.md dev/active/<experiment-name>/plan.md
cp dev/templates/context.md dev/active/<experiment-name>/context.md
cp dev/templates/tasks.md dev/active/<experiment-name>/tasks.md
```

## 2. Plan

Fill in `plan.md`:
- What are we testing?
- What's our hypothesis?
- What metrics will we measure?
- What's the baseline to compare against?

## 3. Baseline

Always measure before changing anything:
```bash
# Run baseline benchmark
uv run python scripts/benchmark.py --config baseline.json
```

Log to `experiments.jsonl`:
```json
{"experiment": "name", "variant": "baseline", "tok_s": 0, "peak_rss_mb": 0, "perplexity": 0, "timestamp": "ISO8601"}
```

## 4. Implement & Measure

- Make one change at a time
- Benchmark after each change
- Log every result, even failures

## 5. Document Findings

Update `context.md` with:
- What worked / what didn't
- Actual numbers vs expectations
- Next steps or dead ends

## 6. Archive

```bash
mv dev/active/<experiment-name> dev/completed/
```
