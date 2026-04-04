# Context: Expert Routing Prediction + Prefetch (H2)

**Issue**: #22

## Current State

All three phases implemented and run on pilot data (211 tokens, 4 prompts). Full collection (Phase 1b) not yet run. Code review completed, all high/medium findings addressed.

## Key Findings

### Phase 1: Routing Pattern Analysis (pilot, 211 tokens)
- **Cross-layer correlation (Jaccard)**: L=1: 0.033, L=2: 0.030, L=3: 0.028 — very low raw overlap between layers
- **Temporal locality**: 0.391 mean recall (consecutive tokens share ~39% of experts)
- **Previous-layer recall**: 0.061 — far below 78.8% prior art. This may indicate Qwen3-30B-A3B has fundamentally different routing patterns from models studied in prior work, or could be a small-sample artifact.
- **Expert frequency**: Gini=low, all 128 experts active, near-uniform distribution (not Zipf-like)
- **Quality check**: PASSED (traced vs untraced outputs match)
- **Speed**: ~0.1 tok/s (memory pressure from 24GB model + traces in 24GB unified memory)

### Phase 2: Predictor Training (pilot data only)
- **L=2 recall@8**: 0.242 — gate FAILED (expected with only 211 training tokens)
- **Predictor latency**: 0.205ms per token (9.8ms for 48 layers) — fast enough for inline use
- **Note**: Insufficient data for meaningful predictor training; full Phase 1b collection needed

### Phase 3: Prefetch Simulation
- **I/O bandwidth**: 2.8 GB/s (pread p50: 0.782ms on synthetic corpus)
- **Throughput ceiling**: Compute-only 10.1 tok/s, Streamed 6.5 tok/s
- **Oracle (perfect prediction)**: 95.8% hit rate, 2.3% stall — meets both gates
- **Predicted (prev-layer heuristic)**: 5.9% hit rate, 34.4% stall — fails gates
- **Key insight**: Oracle ceiling confirms prefetch mechanism works; low hit rate from prev-layer heuristic aligns with the 6.1% previous-layer recall from Phase 1

## Benchmark Summary

| Metric | Baseline (H0) | H2 Oracle | H2 Predicted | Notes |
|--------|---------------|-----------|-------------|-------|
| tok/s  | 6-20 (SSD) | 9.9 (sim) | 6.7 (sim) | Simulation upper bounds |
| prefetch_hit_rate | — | 95.8% | 5.9% | Oracle meets 85% gate |
| pipeline_stall_pct | — | 2.3% | 34.4% | Oracle meets 15% gate |
| predictor_latency | — | — | 0.2ms/tok | Fast enough for inline |
| peak_rss_mb | — | — | — | Measured during runs |

## Review Stats

- Plan review rounds: 8
- Code review rounds: 6
- Total findings addressed: 20 (5 HIGH round 1, 1 HIGH round 2, 3 HIGH round 3, 7 HIGH+MEDIUM round 4, 2 MEDIUM round 5, 2 MEDIUM round 6)

## Next Steps

1. Run full Phase 1b collection (~10,000+ tokens across 20 prompts) for meaningful predictor training
2. Retrain Phase 2 predictors on full data — determine if recall@8 ≥ 90% is achievable
3. If recall remains low, investigate model-specific routing patterns vs prior art assumptions
4. Phase 3 needs C/Metal expert-addressable loading (#17) for real throughput validation
