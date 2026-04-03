# Context: Expert Routing Prediction + Prefetch (H2)

**Issue**: #22

## Current State

Starting Phase 1: Routing pattern analysis. Branch created, experiment docs written, plan under review.

## Key Findings

(None yet — experiment not started)

## Benchmark Summary

| Metric | Baseline (H0) | Current | Delta |
|--------|---------------|---------|-------|
| tok/s  | 6-20 (SSD offload) | — | — |
| peak_rss_mb | — | — | — |
| prediction_accuracy | — | — | — |
| prefetch_hit_rate | — | — | — |

## Review Stats

- Plan review rounds: 0
- Code review rounds: 0
- Total findings addressed: 0

## Blockers

None currently. H0 Phase 4a data is available for baseline measurements.

## Next Steps

1. Complete plan review loop
2. Implement Phase 1: routing_trace.py
3. Run Phase 1 on Qwen3-30B-A3B 4-bit
