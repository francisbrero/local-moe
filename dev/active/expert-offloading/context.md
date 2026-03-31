# Context: Expert Offloading / SSD Streaming

**Issue**: #2

## Current State

All benchmarks (Phases 0-4a) complete. Results strongly support expert offloading viability on M4 Pro. Phase 4b (C/Metal staging buffers) recommended as next step.

## Key Findings

### Phase 0: Checkpoint Audit
- **Target model**: Qwen3-30B-A3B (128 experts/layer, K=8, 48 layers)
- Non-expert weights: only 0.5 GB at 4-bit — fits easily in memory
- Expert size at 4-bit: ~2.25 MB each
- Machine is 24GB M4 Pro (not 16GB as initially assumed)

### Phase 1: NVMe Profiling
- M4 Pro NVMe achieves **15-17 GB/s** sequential read (far exceeds 4 GB/s target)
- pread warm cache: 15-17 GB/s across all chunk sizes
- pread F_NOCACHE (cold): 5.5-6.5 GB/s
- mmap sequential: 14-17 GB/s (warm)
- mmap random: 9-11 GB/s
- Scattered reads: 5-7 GB/s

### Phase 2: Page Cache Behavior
- pread default: p50=0.31ms, 5.6 GB/s — best overall throughput
- pread F_NOCACHE: p50=0.66ms, 5.4 GB/s — bypasses cache, consistent latency
- mmap MADV_RANDOM: p50=0.25ms but p95=14.8ms, 77.5% cache residency — best for hot experts
- mmap default/sequential: high p95 tail latency (9-11ms), ~63-67% residency
- Zipf access pattern achieves 63-78% cache residency with 4GB corpus
- pread preferred for predictable latency; mmap viable for cache-friendly patterns

### Phase 3: GPU/SSD Contention
- Only **0.2% GPU degradation** with concurrent SSD access on M4 Pro
- Concurrent pipeline viable — no contention between GPU compute and SSD reads
- Overlap efficiency: ~100% (GPU and SSD can run fully in parallel)

### Phase 4a: Synthetic Expert Streaming
- **Streamed is faster than in-memory**: 0.83x ratio (176.7ms vs 212.0ms for 384 expert calls)
- Expert call latency: p50=0.40ms, p95=0.68ms (streamed)
- **Zero-copy confirmed broken**: MLX copies mmap data internally (229.9 MB RSS growth over 100 iterations)
- Prototype viability: **PASS** (well within 3x threshold)
- Recommended Phase 4b path: C/Metal staging buffers for true zero-copy

## Benchmark Summary

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| NVMe bandwidth | 15-17 GB/s (warm), 5.5-6.5 GB/s (cold) | >4 GB/s | PASS |
| GPU/SSD contention | 0.2% degradation | <10% | PASS |
| Expert call latency (p50) | 0.40 ms | <5 ms | PASS |
| Expert call latency (p95) | 0.68 ms | <10 ms | PASS |
| Streaming overhead | 0.83x (faster) | <3x | PASS |
| Cache residency (Zipf) | 63-78% | >50% | PASS |
| Zero-copy (MLX mmap) | NO | — | Phase 4b needed |

## Review Stats

- Plan review rounds: 7
- Code review rounds: 2
- Total findings addressed: 24 (20 plan + 4 code)
- Runtime bug fixes: 2 (BufferError in zero-copy test, ValueError reshape in GEMM)

## Blockers

None.

## Next Steps

1. Promote to Phase 4b: Implement C/Metal staging buffers for true zero-copy expert loading
2. Test with actual Qwen3-30B-A3B model weights (not synthetic data)
3. Integrate with MLX inference pipeline
4. Run under memory pressure to validate with constrained memory (16GB scenario)
