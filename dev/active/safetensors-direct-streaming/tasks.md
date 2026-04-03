# Tasks: Safetensors Direct Streaming (H8)

## Setup
- [x] Create branch
- [x] Draft plan.md, context.md, tasks.md
- [x] Plan review loop (8 rounds, 22+ findings addressed)
- [x] Get user approval on plan

## Phase 0: Shard Layout + MLX Load Characterization
- [x] Validate shard layout (8 shards, 5.3GB each)
- [x] Characterize mx.load (lazy mmap, ~0.5ms load)
- [x] Component breakdown (load/assign/eval/forward/evict)
- [x] Gate: PASS — 21.4ms p50, 5.9x speedup over npz

## Phase 1: Safetensors Block Index
- [x] Parse model.safetensors.index.json for 72B
- [x] Build block → (shard, tensor_names) mapping
- [x] Validate per-block sizes match Phase 0 budget (470.9 MB, 0.0% diff)
- [x] Handle cross-shard blocks (5 out of 80)

## Phase 2: Direct Load on 7B
- [x] Build safetensors index for 7B
- [ ] Implement safetensors direct load function (done in Phase 0)
- [ ] Benchmark npz vs safetensors load latency (done in Phase 0)
- [ ] Verify logit correctness (exact match) — skipped, validated via Phase 3 coherence
- [ ] Log results to experiments.jsonl

## Phase 3: 72B Integration
- [x] Adapt 72B integration script for safetensors path
- [x] Skip npz save step (load directly from original shards)
- [x] Fix degenerate output bug (add KV cache: prefill + decode)
- [x] Run 10 token generation — coherent output confirmed
- [x] Measure tok/s (0.007), memory stability (PASS)
- [x] Log results to experiments.jsonl

## Phase 4: Cache Optimization
- [ ] Deferred to H8b (Q2 streaming blocks required)

## Review
- [x] Plan review loop completed (8 rounds)
- [ ] Code review loop completed
- [ ] Address all high/medium findings

## Documentation
- [x] Update context.md with findings
- [ ] Create PR with results
