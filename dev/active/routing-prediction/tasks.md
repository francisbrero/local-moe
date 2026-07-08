# Tasks: Expert Routing Prediction + Prefetch (H2)

## Setup
- [x] Create branch (experiment/routing-prediction)
- [x] Create experiment docs
- [x] Plan review loop (8 rounds)

## Phase 1: Routing Pattern Analysis
- [x] Implement routing_trace.py (MLX model hook + trace capture)
- [x] Run on Qwen3-30B-A3B 4-bit (pilot: 4 prompts, 211 tokens)
- [x] Analyze cross-layer correlation
- [x] Analyze temporal locality
- [x] Analyze expert co-occurrence
- [x] Log results to experiments.jsonl
- [x] Compute previous-layer expert recall baseline
- [x] Report descriptive metrics (no hard gate)
- [ ] Run full Phase 1b collection (20 prompts, 10K+ tokens)

## Phase 2: Predictor Design
- [x] Implement routing_predictor.py
- [x] Train cross-layer linear predictor (2 layers, pre-attention input)
- [x] Evaluate accuracy at L=1, 2, 3 lookahead (pilot only)
- [x] Build per-layer co-activation buddy table
- [x] Log results to experiments.jsonl
- [ ] Gate check: mean recall@8 ≥ 90% at L=2 (needs full data)

## Phase 3: Async Prefetch Pipeline
- [x] Implement routing_prefetch_bench.py
- [x] Background thread prefetch simulation with timing model
- [x] Staging buffer + LRU eviction
- [x] Measure prefetch hit rate and throughput
- [x] Compare: no-prefetch vs oracle vs predicted
- [x] Log results to experiments.jsonl
- [ ] Gate check on predicted run (needs trained predictor with full data)

## Validation
- [x] All metrics logged to experiments.jsonl
- [x] Quality sanity check (traced vs untraced outputs match)

## Review
- [x] Plan review loop completed (8 rounds)
- [x] Code review loop completed (6 rounds, 20 findings addressed)

## Documentation
- [x] Update context.md with findings
- [ ] Archive to dev/completed/ when experiment fully done
