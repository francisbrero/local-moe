# Context: Expert Routing Prediction + Prefetch (H2)

**Issue**: #22
**Status**: Experiment complete — negative result for Qwen3-30B-A3B routing prediction

## Conclusion

Expert routing prediction viability is **model-dependent**. Qwen3-30B-A3B has near-uniform expert routing (Gini=0.14, all 128/128 experts active) with very low cross-layer correlation (Jaccard ~0.03), making prediction fundamentally harder than the models studied in prior art (DeepSeek V2, Phi-MoE) which reported 93-97% recall. The prefetch *mechanism* works perfectly (oracle: 95.8% hit rate), but no predictor we tested can achieve the 90% recall needed to make it useful on this model.

This does not invalidate the approach — it means models with concentrated/Zipf-like routing (e.g., DeepSeek V2) would benefit from this technique, while models with uniform routing (Qwen3-30B-A3B) would not.

## Key Findings

### Phase 1b: Routing Pattern Analysis (1582 tokens, 19 prompts)
- **Cross-layer correlation (Jaccard)**: L=1: 0.034, L=2: 0.032, L=3: 0.033 — very low
- **Temporal locality**: 0.405 mean recall (consecutive tokens share ~40% of experts)
- **Previous-layer recall**: 0.062 — **far below 78.8% prior art** (confirmed across 1582 tokens, not a small-sample artifact)
- **Expert frequency**: Gini=0.139, all 128 experts active, near-uniform distribution (NOT Zipf-like)
- **Quality check**: PASSED (traced vs untraced outputs match)

### Phase 2: Predictor Training (1582 tokens)
- **L=1 recall@8**: 0.628
- **L=2 recall@8**: 0.625 — **gate FAILED** (needs ≥0.90)
- **L=3 recall@8**: 0.610
- **Exact set match**: ~3.5%
- **Predictor latency**: 0.189ms/token (9.1ms for 48 layers) — fast enough for inline use
- Significant improvement over pilot (0.242 → 0.625) but plateaus well below gate

### Phase 3: Prefetch Simulation
- **Oracle (perfect prediction)**: 95.8% hit rate, 2.3% stall, 9.9 tok/s — **mechanism validated**
- **Predicted (prev-layer heuristic)**: 5.6% hit rate, 34.5% stall, 6.6 tok/s — fails gates
- **Throughput ceiling**: Compute-only 10.1 tok/s (MLX Python path limitation)
- **I/O bandwidth**: 2.9 GB/s (pread p50: 0.767ms)

## Benchmark Summary

| Metric | Baseline (H0) | H2 Oracle | H2 Predicted | Notes |
|--------|---------------|-----------|-------------|-------|
| tok/s  | 6-20 (SSD) | 9.9 (sim) | 6.6 (sim) | Simulation upper bounds |
| prefetch_hit_rate | — | 95.8% | 5.6% | Oracle meets 85% gate |
| pipeline_stall_pct | — | 2.3% | 34.5% | Oracle meets 15% gate |
| recall@8 (L=2) | — | 100% | 62.5% | Predictor below 90% gate |
| predictor_latency | — | — | 0.189ms/tok | Fast enough for inline |

## Why Prior Art Numbers Don't Apply

Prior art (ETH Zurich, Fate) reports 93-97% recall on DeepSeek V2 Lite, Phi-MoE, and similar models. Those models have:
- Concentrated/Zipf-like expert usage (some experts used far more than others)
- Strong cross-layer correlation (same experts tend to activate across layers)
- Previous-layer recall of ~78.8% as a cheap baseline

Qwen3-30B-A3B has the opposite pattern:
- Near-uniform expert usage (Gini=0.14 vs >0.3 for Zipf)
- Very low cross-layer correlation (Jaccard ~0.03)
- Previous-layer recall of only 6.2%

This suggests the routing architecture (gate design, expert count, training procedure) determines predictability, not model size or MoE structure alone.

## Review Stats

- Plan review rounds: 8
- Code review rounds: 6
- Total findings addressed: 20 (8 HIGH, 12 MEDIUM)

## Deliverables

1. `scripts/routing_trace.py` — MLX tracing hooks for expert routing capture (reusable for other models)
2. `scripts/routing_predictor.py` — Per-layer linear predictor training pipeline
3. `scripts/routing_prefetch_bench.py` — Synthetic prefetch pipeline simulation with timing model
4. Per-layer co-activation buddy tables in `routing_traces/buddy_table.json`
5. Routing traces for 1582 tokens across 19 diverse prompt categories

## Recommendations

1. **Try a model with concentrated routing** (e.g., DeepSeek V2 Lite) to validate the approach works where prior art says it should
2. **For Qwen3-30B-A3B specifically**: routing prediction won't help; focus on other latency-hiding strategies (larger page cache, smarter eviction, compute overlap)
3. **The prefetch pipeline code is reusable** — when combined with a model that has predictable routing, the mechanism is sound
