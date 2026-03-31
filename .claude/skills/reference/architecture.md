---
description: Project architecture and research approach overview
globs: []
alwaysApply: false
---

# Architecture Reference

## Project Structure

```
local-moe/
├── CLAUDE.md              # AI agent instructions
├── Agents.md              # Research approaches (10 approaches)
├── experiments.jsonl       # Structured experiment log
├── src/
│   ├── metal/             # Metal compute shaders (.metal)
│   ├── *.c / *.h          # C inference engine
│   └── ...
├── scripts/
│   ├── benchmark.py       # Benchmarking harness
│   └── ...
├── dev/
│   ├── active/            # Current experiments (gitignored)
│   ├── completed/         # Archived experiments (gitignored)
│   └── templates/         # Experiment doc templates
├── models/                # Downloaded models (gitignored)
└── .claude/
    ├── commands/          # Slash commands
    ├── hooks/             # Automation hooks
    ├── skills/            # Context skills
    └── settings.local.json
```

## Research Phases

1. **Baseline & Quick Wins** — llama.cpp MoE offloading, TurboQuant, extreme quant, MLX benchmarks
2. **Custom Streaming Pipeline** — Flash MOE Metal port, HOBBIT mixed precision, ML cache, fused shaders
3. **Advanced Optimizations** — Speculative decoding, layer streaming, BitNet, M4-specific profiling
4. **Integration** — Unified engine, benchmark harness, documentation

## Key Metrics

| Metric | Description |
|--------|-------------|
| tok/s | Tokens per second (decode) |
| ttft | Time to first token (prefill) |
| peak_rss_mb | Peak resident memory |
| cache_hit_rate | Expert cache hit rate |
| perplexity | Output quality |

## GitHub Issues

All research approaches are tracked as GitHub issues with `research` + priority labels.
Use `gh issue list --label research` to see all approaches.
