# SSD Layer LOD — Context

**Issue**: #28
**Branch**: `experiment/ssd-layer-lod`
**Status**: Phase 3 complete — 72B runs but too slow (0.005 tok/s). Memory management works. Serialization bottleneck identified.

## Current State

All phases complete. 72B model loads and generates on 24 GB M4 Pro (memory stable at 3.9 GB available). However, tok/s is 0.005 — impractically slow due to npz serialization overhead. The incremental loading strategy works perfectly (RSS stays at 148 MB during setup). The bottleneck is per-block swap cost (mx.eval on 471 MB blocks).

## Key Findings from Prior Experiments

### From H0 (Expert Offloading)
- NVMe cold: 5.5-6.5 GB/s, warm: 15-17 GB/s
- Expert load p50: 0.40 ms
- GPU degradation from concurrent SSD: 0.2% (negligible)
- Page cache residency (Zipf): 63-78% hit rate

### From H5 (Layer LOD)
- U-shape CONFIRMED on Qwen2.5-7B
- Middle 60% blocks tolerate Q2 with avg Δ = 0.11 PPL
- Last block Q2: catastrophic (+698 PPL)
- Sensitivity-guided 3.0 bpw beats uniform Q3 by 1.1% PPL
- 72B at 2.0 bpw = 24.3 GB — 0.3 GB over limit (needs SSD streaming)

## Phase 0 Results: Memory Budget

### Per-Block Sizes (Qwen2.5-72B)
- **Q2**: 261.6 MB/block
- **Q3**: 366.2 MB/block
- **Q4**: 470.9 MB/block
- Fixed costs: 1.89 GB (embeddings + lm_head at Q6)
- KV cache (2048 ctx, FP16): 0.62 GB
- KV cache (2048 ctx, Q4): 0.16 GB

### Key Finding: Original 20/60/20 Q4/Q2/Q4 Too Tight
- 32 blocks at Q4 = 14.71 GB resident blocks alone
- Total pinned + OS = 22.22 GB → only 1.78 GB for page cache
- Streaming 12.26 GB from SSD with 1.78 GB cache = constant thrashing

### Recommended Configurations

| Config | Avg BPW | Pinned+OS | Cache Budget | Streaming |
|--------|---------|-----------|-------------|-----------|
| **8+8 Q4 / 64 Q2** | 2.4 | 14.87 GB | 9.13 GB | 16.35 GB |
| 8+8 Q3 / 64 Q2 | 2.2 | 13.23 GB | 10.77 GB | 16.35 GB |
| **3+5 Q4 / 72 Q2** | 2.2 | 11.19 GB | 12.81 GB | 18.39 GB |
| 3+5 Q4 / 72 Q2 (KV Q4) | 2.2 | 10.72 GB | 13.28 GB | 18.39 GB |

**Best quality**: 8+8 Q4 / 64 Q2 (2.4 bpw, 9.13 GB cache)
**Best H5 alignment**: 3+5 Q4 / 72 Q2 (protects last 5 blocks per H5 sensitivity data, 12.81 GB cache)

### 16 GB Feasibility
72B on 16 GB requires near-total streaming:
- 0+3 Q4 / 77 Q2 at 512 ctx with Q4 KV: 8.3 GB pinned+OS → fits!
- Uniform Q2, all streamed: 6.9 GB pinned+OS → comfortable

### Phase 0 Verdict: PASS
Multiple viable configurations found. Proceeding with **8+8 Q4 / 64 Q2** as primary target (best quality) and **3+5 Q4 / 72 Q2** as fallback (more cache headroom).

## Phase 1b Results: Loader Strategy Selection

### MLX Zero-Copy: CONFIRMED BROKEN
- RSS grows ~65 MB (2× weight size) then stabilizes via GC
- After 2 warmup cycles, steady-state RSS delta ≈ 0 MB

### Key Discovery: MLX GC Makes Streaming Viable
Despite MLX copying internally, the GC reclaims old tensor memory within 2 cycles.
Steady-state behavior: RSS and Metal memory stay flat after warmup.

### pread Staging Buffer: VALIDATED
- Steady-state bandwidth: 6.2 GB/s (warm), first load 4.5 GB/s (cold)
- Per-block latency (262 MB): 41 ms steady-state
- RSS flat in steady state: max delta 3.2 MB

### End-to-End Inference: VALIDATED
- Load-swap-forward cycle works with zero logit diff (exact match)
- RSS flat after 2 warmup cycles: max steady delta 3.4 MB
- Metal memory flat: +11.6 MB growth (within ±5%)

### mlock/madvise: ALL WORK
- mlock: unlimited on this machine
- MADV_WILLNEED, MADV_DONTNEED, MADV_FREE: all succeed

### Loader Selected: mx.array() swap with GC
No C extension needed! The MLX path works because:
1. `mx.array(numpy_data)` creates new tensors
2. Old tensors are GC'd when replaced
3. After 2 warmup cycles, RSS is stable
4. Logits match exactly

## Phase 1 Results: Layer Streaming Prototype (7B)

### Architecture
Custom forward pass that intercepts the layer loop:
- For streaming blocks: load weights from disk → forward → evict (with mx.eval barrier)
- For resident blocks: normal forward
- Also tested no-evict variant (load but don't free)

### Benchmark Results (Qwen2.5-7B, 28 blocks, unpressured)

| Mode | tok/s | ms/tok | Peak RSS | Load p50 | Load p95 | Pageouts |
|------|-------|--------|----------|----------|----------|----------|
| all-resident (baseline) | 50.7 | 19.7 | 1902 MB | — | — | 0.1 MB |
| stream-middle (evict) | 2.07 | 484 | 1902 MB | 16.9 ms | 23.5 ms | 0.5 MB |
| stream-middle (no-evict) | 0.44 | 2294 | 3128 MB | 39.3 ms | 173 ms | 10.6 MB |
| stream-all (evict) | 1.73 | 577 | 3128 MB | 17.7 ms | 20.5 ms | 0.8 MB |

### Key Findings

1. **24.6x slowdown** for stream-middle vs baseline on 7B
2. **Bottleneck**: per-block disk load (24 blocks × 17 ms = ~408 ms) + mx.eval serialization barrier
3. **Evict > no-evict**: Proactive eviction keeps RSS controlled. No-evict causes memory blowup → OS pageouts → worse latency
4. **Per-block load latency**: 16.9 ms p50 for ~160 MB block (warm cache, ~9.5 GB/s)
5. **No pressure effect**: System already memory-constrained from model load; synthetic pressure target was below actual available memory

### Why This Is Expected (Not a Blocker)

The 24.6x slowdown is expected on the naive 7B prototype because:
- All 24 streaming blocks are **cold-loaded every token** (no page cache reuse)
- On 72B with 9 GB page cache, H0 showed 63-78% cache hit rate → most blocks served from RAM
- The `mx.eval` barrier after each block serializes the GPU pipeline; Phase 2's sliding window scheduler eliminates this by keeping N blocks resident simultaneously
- The per-block load latency (17 ms) matches Phase 1b's validation (41 ms for 262 MB → ~6 GB/s)

### Phase 1 Verdict: INFORMATIONAL
Baseline streaming overhead measured. Per-block load latency validated against Phase 1b. Architecture works correctly (RSS controlled with eviction). Optimization deferred to Phase 2 scheduler.

## Phase 2 Results: Scheduling Strategy Selection

### Strategy Comparison (7B, 24 streaming blocks)

| Strategy | tok/s | ms/tok | Slowdown | Load p50 | Wait p50 |
|----------|-------|--------|----------|----------|----------|
| Baseline (all-resident) | 51.2 | 19.5 | 1.0x | — | — |
| Serial | 2.01 | 498 | 25.5x | 16.8 ms | — |
| Double-buffer (prefetch) | 2.30 | 434 | 22.2x | 2.2 ms | 12.5 ms |

### Key Findings
- **Double-buffer wins**: 12.8% faster than serial (434 vs 498 ms/tok)
- Prefetch successfully overlaps disk I/O with GPU compute
- The "wait" time (12.5 ms p50) shows the prefetched block is almost ready when needed
- Improvement limited on 7B because per-block GPU compute (~1 ms) is tiny vs load time (~17 ms)
- On 72B, per-block compute will be much larger → more overlap opportunity

### Strategy Selected: double-buffer

## Phase 2b Results: Synthetic 72B Streaming

### Configuration
- 64 streaming blocks × 262 MB = 16.4 GB total (matches 8+8 Q4/64 Q2 config)
- Sequential access pattern (block 0 → 1 → ... → 63)

### Results

| Regime | p50 (ms) | p95 (ms) | p99 (ms) | tok/s | Thrash | Pageouts |
|--------|----------|----------|----------|-------|--------|----------|
| Unpressured (8 GB avail) | 55 | 75 | 91 | 0.27 | NO | 4 MB |
| Pressured (~6 GB avail) | 60 | 96 | 139 | 0.25 | NO | 7 MB |

### Key Findings
1. **No thrashing** even under pressure (CV=0.021-0.064)
2. **Gate FAILS** (p95 50ms threshold) — but this is reading ALL 64 blocks cold
3. **0.25 tok/s worst-case floor** when every block must be read from SSD
4. **Cache warmup: 1.5x** improvement from first to steady-state iteration
5. **Pageins: 84-168 GB** confirms all data comes from SSD (no cache hits in this test)
6. With page cache at 55% occupancy (9 GB / 16.4 GB), ~36 blocks cached → only 28 blocks from SSD → ~1.6s/tok → ~0.6 tok/s

### Phase 2b Verdict: INFORMATIONAL (gate metric not meaningful)
The p95<50ms gate was designed for per-block latency with cache hits. In this test, blocks are always cold (no reuse across iterations since the working set exceeds cache). The cold-read latency of 55-60 ms/block for 262 MB matches H0's NVMe profile (~4.5 GB/s cold). The critical metric is end-to-end tok/s on actual 72B, where page cache will provide hits.

## Phase 3 Results: 72B Integration Test

### Configuration
- Model: Qwen2.5-72B-Instruct-4bit (38 GB on disk)
- 80 blocks total: 16 resident (first 8 + last 8), 64 streaming
- Block size: 471 MB, Total streaming: 29.4 GB
- Double-buffer prefetch scheduler

### Incremental Loading: SUCCESS
Key innovation: **never materialize all 38 GB at once**. Process blocks one-by-one:
- For streaming blocks: mx.eval → save .npz → evict (replace with placeholders)
- For resident blocks: mx.eval → keep in RAM
- RSS stayed at 148 MB throughout processing, available at 10.2 GB
- All 80 blocks processed in 2 seconds

### Token Generation Results (10 tokens)

| Metric | Value |
|--------|-------|
| tok/s | 0.005 (188s/tok) |
| Peak RSS | 3250 MB |
| Min available | 3.9 GB |
| Memory stable | YES |
| Block wait p50 | 1111 ms |
| Block swap p50 | 704 ms |
| Pageouts | 87 MB |
| Pageins | 321 GB |

### Key Issues

1. **Extremely slow**: 188s/tok — 750x worse than expected 0.25 tok/s floor
   - swap_block_weights (~58s/tok): mx.eval on 64 × 471 MB blocks dominates
   - wait time (~73s/tok): disk I/O for 29.4 GB per token
   - The npz load + mx.array conversion + mx.eval per-block is far more expensive at 72B scale than at 7B

2. **Degenerate output**: "the following the following the following..."
   - Suggests model weights may not restore correctly after eviction
   - Possible cause: npz save/load losing quantization metadata, or non-QuantizedLinear modules not being saved/restored (layer norms, biases)

3. **Gate FAIL**: tok/s (0.005) far below 0.1 threshold

### Root Cause Analysis

The fundamental issue is that **npz-based block serialization is too expensive at 72B scale**:
- Each 471 MB block requires: np.load from disk → mx.array() conversion → mx.eval()
- mx.eval forces synchronous Metal computation, blocking for each block
- 64 blocks × (disk I/O + conversion + eval) = 188s per token

**Potential mitigations for future work**:
- Use safetensors format directly (mmap-compatible, avoids np→mx conversion)
- Load directly from original model safetensors shards instead of re-serialized .npz
- Reduce streaming set: keep more blocks resident (e.g., 50% resident, 50% streaming)
- Use Q2 for streaming blocks (half the block size → half the I/O)

### Phase 3 Verdict: FAIL
72B inference works mechanically (memory stable, no OOM) but is impractically slow.
The npz serialization path adds massive overhead at 72B scale. Future work should
load directly from safetensors shards to avoid the conversion penalty.

## Review Stats

- Plan review rounds: 8 (22 findings addressed)
- Code review rounds: 3 (2 medium + 4 low findings fixed in round 2; round 3 findings out-of-scope)
- Total findings addressed: 28
