# Tasks: Expert Routing Prediction + Prefetch (H2)

## Setup
- [x] Create branch (experiment/routing-prediction)
- [x] Create experiment docs
- [ ] Plan review loop

## Phase 1: Routing Pattern Analysis
- [ ] Implement routing_trace.py (MLX model hook + trace capture)
- [ ] Run on Qwen3-30B-A3B 4-bit with diverse prompts
- [ ] Analyze cross-layer correlation
- [ ] Analyze temporal locality
- [ ] Analyze expert co-occurrence
- [ ] Log results to experiments.jsonl
- [ ] Compute previous-layer expert recall baseline
- [ ] Report descriptive metrics (no hard gate)

## Phase 2: Predictor Design
- [ ] Implement routing_predictor.py
- [ ] Train cross-layer linear predictor (2 layers, pre-attention input)
- [ ] Evaluate accuracy at L=1, 2, 3 lookahead
- [ ] Build co-activation buddy table
- [ ] Log results to experiments.jsonl
- [ ] Gate check: mean recall@8 ≥ 90% at L=2 (95% CI width ≤ 5pp)

## Phase 3: Async Prefetch Pipeline
- [ ] Implement routing_prefetch_bench.py
- [ ] Background thread prefetch simulation
- [ ] Staging buffer + Least-Stale eviction
- [ ] Measure prefetch hit rate and throughput
- [ ] Compare: no-prefetch vs oracle vs predicted
- [ ] Log results to experiments.jsonl
- [ ] Gate check: hit rate ≥ 85%, pipeline stall ≤ 15% (tok/s is informational upper bound only)

## Validation
- [ ] All metrics logged to experiments.jsonl
- [ ] Results compared against H0 baseline

## Review
- [ ] Plan review loop completed
- [ ] Code review loop completed
- [ ] Address all high/medium findings

## Documentation
- [ ] Update context.md with findings
- [ ] Archive to dev/completed/ when done
