# Context: Expert Offloading / SSD Streaming

**Issue**: #2

## Current State

Implementation complete for Phases 0-4a. All scripts written, reviewed, and fixes applied. Ready for benchmarking.

Scripts implemented:
- `scripts/experiment_utils.py` — Shared utilities (logging, memory pressure, metrics)
- `scripts/checkpoint_audit.py` — Phase 0: MoE checkpoint size audit
- `scripts/nvme_profile.py` — Phase 1: NVMe read profiling
- `scripts/page_cache_bench.py` — Phase 2: Page cache behavior with Zipf patterns
- `scripts/gpu_ssd_contention.py` — Phase 3: Serial vs concurrent GPU/SSD
- `scripts/expert_stream_synthetic.py` — Phase 4a: Synthetic expert streaming

## Key Findings

None yet — benchmarks not yet run. Scripts are ready for execution.

## Benchmark Summary

| Metric | Baseline | Current | Delta |
|--------|----------|---------|-------|
| tok/s  |          |         |       |
| peak_rss_mb |     |         |       |
| gpu_memory_mb |   |         |       |
| ssd_bandwidth_gbs | |       |       |
| cache_hit_rate |   |         |       |

## Review Stats

- Plan review rounds: 7
- Code review rounds: 2
- Total findings addressed: 24 (20 plan + 4 code)

## Blockers

None currently.

## Next Steps

1. Run Phase 0: `uv run python scripts/checkpoint_audit.py`
2. Run Phase 1: `uv run python scripts/nvme_profile.py`
3. Run Phase 2: `uv run python scripts/page_cache_bench.py --corpus-size-gb 20 --pressure 10`
4. Run Phase 3: `uv run python scripts/gpu_ssd_contention.py`
5. Run Phase 4a: `uv run python scripts/expert_stream_synthetic.py`
6. Evaluate promotion criteria for Phase 4b
