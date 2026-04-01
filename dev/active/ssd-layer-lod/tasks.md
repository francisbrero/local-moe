# SSD Layer LOD — Tasks

## Phase 0: Memory Budget Modeling
- [ ] Download Qwen2.5-72B config.json and safetensors index
- [ ] Compute per-block sizes at Q2/Q3/Q4
- [ ] Model resident set (first/last 20% at Q4 + embeddings + KV cache)
- [ ] Model streaming set (middle 60% at Q2)
- [ ] Verify resident set fits in ≤18 GB
- [ ] Log results to experiments.jsonl

## Phase 1: Layer Streaming Prototype (7B)
- [ ] Implement mmap-based layer loading wrapper
- [ ] Measure all-resident baseline (tok/s, RSS)
- [ ] Implement stream-middle-only mode
- [ ] Measure stream-middle vs all-resident
- [ ] Measure page cache hit rates over 100 tokens
- [ ] Log results to experiments.jsonl

## Phase 2: Prefetch Pipeline
- [ ] Implement double-buffer with madvise(MADV_WILLNEED)
- [ ] Measure overlap efficiency
- [ ] Context length sweep: 128, 512, 2048 tokens
- [ ] Log results to experiments.jsonl

## Phase 3: 72B Integration Test
- [ ] Download/prepare Qwen2.5-72B mixed precision
- [ ] Run inference with resident + streaming split
- [ ] Measure tok/s, peak RSS, perplexity
- [ ] Compare against Qwen2.5-14B Q4 baseline
- [ ] Memory stability test (500 tokens)
- [ ] Log results to experiments.jsonl

## Phase 4: Optimization
- [ ] Tune resident/streaming split with sensitivity data
- [ ] Test with KV cache compression
- [ ] Profile Metal GPU utilization
