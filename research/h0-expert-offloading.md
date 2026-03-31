# H0: Expert Offloading / SSD Streaming

**Status**: Promising
**Analogy**: Virtual memory / demand paging
**Bottleneck addressed**: Memory capacity
**Issue**: #2 (Phases 0-4a complete), #17 (Phase 4b planned)

## The Insight

Operating systems solved the "program larger than RAM" problem decades ago with virtual memory: keep the working set in RAM, page the rest to disk, and trust locality of reference to keep hit rates high.

MoE models have the same structure. A 30B MoE like Qwen3-30B-A3B has 128 experts per layer, but only activates 8 per token. That's 6,144 total expert blocks, but only 384 active at any moment. The inactive 93.75% of expert weights don't need to be in RAM — they can live on SSD and stream in on demand.

## Hypothesis

> If we keep only non-expert weights (~0.5 GB) and the active expert set in RAM, streaming the rest from NVMe SSD, we can run 30B+ MoE models on 16GB machines that otherwise top out at ~7-8B dense models.

## Mechanism

```
For each token:
  1. Router selects K active experts per layer
  2. For each active expert:
     a. If expert is in page cache → near-instant access
     b. If not → stream from NVMe SSD (mmap or pread)
  3. Dequantize expert weights (4-bit → float16)
  4. Run expert GEMM
  5. Combine expert outputs via router weights
```

The OS page cache acts as an automatic LRU cache. Popular experts (Zipf-distributed) stay resident; cold experts get evicted and re-loaded on demand.

## Results (Phases 0-4a)

### Phase 0: Checkpoint Audit

Target model: **Qwen3-30B-A3B** (30B params, 3B active per token)
- 128 experts/layer, K=8 active, 48 layers
- Expert size at 4-bit: 2.25 MB each
- Non-expert weights at 4-bit: 0.53 GB
- Total expert weights: 13.5 GB (stored on SSD)
- Active experts per token: 384 × 2.25 MB = 0.84 GB

### Phase 1: NVMe Profiling (M4 Pro)

| Method | Bandwidth |
|--------|-----------|
| pread (warm cache) | 15-17 GB/s |
| pread + F_NOCACHE (cold) | 5.5-6.5 GB/s |
| mmap + MADV_SEQUENTIAL | 14-17 GB/s |
| mmap + MADV_RANDOM | 9-11 GB/s |
| Scattered reads (3 sub-chunks) | 5-7 GB/s |

Key finding: M4 Pro NVMe far exceeds the ~4 GB/s minimum needed for streaming. Even cold reads at 5.5 GB/s can load a 2.25 MB expert in 0.4 ms.

### Phase 2: Page Cache Behavior (Zipf access, 4GB corpus)

| Access Method | p50 (ms) | p95 (ms) | GB/s | Cache Residency |
|---------------|----------|----------|------|-----------------|
| pread default | 0.31 | 1.71 | 5.62 | N/A |
| pread F_NOCACHE | 0.66 | 1.26 | 5.44 | N/A |
| mmap default | 0.43 | 11.04 | 1.11 | 66.9% |
| mmap MADV_SEQUENTIAL | 0.49 | 9.96 | 1.18 | 62.9% |
| mmap MADV_RANDOM | 0.25 | 14.80 | 1.18 | 77.5% |

Key findings:
- Zipf access achieves 63-78% page cache residency — hot experts stay in RAM
- pread gives best predictable throughput (low p95)
- mmap MADV_RANDOM gives best residency but high tail latency

### Phase 3: GPU/SSD Contention

| Metric | Value |
|--------|-------|
| GPU degradation with concurrent SSD | **0.2%** |
| Overlap efficiency | ~100% |

Key finding: On M4 Pro, GPU compute and SSD reads run fully in parallel with negligible contention. This is dramatically better than M2 Ultra (73% degradation). Concurrent pipeline is viable.

### Phase 4a: Synthetic Expert Streaming

| Metric | In-Memory | Streamed | Ratio |
|--------|-----------|----------|-------|
| Latency p50 (ms) | 0.487 | 0.402 | 0.82x |
| Latency p95 (ms) | 0.773 | 0.684 | 0.88x |
| Total time (ms) | 212.0 | 176.7 | 0.83x |

Key findings:
- SSD-streamed GEMM is **17% faster** than in-memory baseline (likely cache effects with small experts)
- MLX copies mmap data internally — `np.frombuffer(mmap)` + `mx.array()` causes full copies (229.9 MB RSS growth over 100 iterations)
- Zero-copy requires C/Metal staging buffers (Phase 4b, #17)
- Prototype viability: **PASS** (well within 3x overhead threshold)

## What This Enables

### Without offloading (standard MLX, everything in RAM)

| Model | Size @ 4-bit | Fits 24GB? | Fits 16GB? | Speed |
|-------|-------------|------------|------------|-------|
| Dense 7B | ~4 GB | Yes | Yes | ~100+ tok/s |
| Dense 14B | ~8 GB | Yes | Yes | ~40-50 tok/s |
| Qwen3-30B-A3B | 14.0 GB | Tight | No | ~60-80 tok/s |
| Mixtral-8x7B | 21.8 GB | No | No | N/A |

### With expert offloading

| Model | RAM needed | On SSD | Fits 16GB? | Speed (est.) |
|-------|-----------|--------|------------|-------------|
| Qwen3-30B-A3B @ 4-bit | ~2 GB | 13.5 GB | **Yes** | ~6-20 tok/s |
| Qwen3-30B-A3B @ 3-bit | ~1.5 GB | 10.1 GB | **Yes** | ~8-25 tok/s |
| Mixtral-8x7B @ 4-bit | ~6.5 GB | 21 GB | **Yes** | ~3-10 tok/s |

### The trade-off

A 30B MoE at ~10 tok/s produces significantly better outputs than a 7B dense model at 100 tok/s. Expert offloading lets a 16GB machine run models that otherwise require 32GB+.

| Machine | Without offloading | With offloading |
|---------|-------------------|----------------|
| 16GB M4 | Max ~7-8B dense | **30B MoE** |
| 24GB M4 Pro | 30B MoE (tight, short context) | **30B+ MoE with full context headroom** |

## Key Numbers

| Metric | Value | Source |
|--------|-------|--------|
| NVMe bandwidth (warm) | 15-17 GB/s | Phase 1 |
| NVMe bandwidth (cold) | 5.5-6.5 GB/s | Phase 1 |
| GPU/SSD contention | 0.2% | Phase 3 |
| Expert load latency (p50) | 0.40 ms | Phase 4a |
| Streaming overhead | 0.83x (faster) | Phase 4a |
| Page cache residency (Zipf) | 63-78% | Phase 2 |
| MLX zero-copy | No (copies internally) | Phase 4a |
| RAM for 30B MoE | ~2 GB | Phase 0 |
| RAM without offloading | 14 GB | Phase 0 |

## Open Questions

1. ~~Is NVMe fast enough for per-token expert streaming?~~ **Yes — 5.5+ GB/s cold, 15+ GB/s warm.**
2. ~~Does GPU compute degrade with concurrent SSD reads?~~ **No — 0.2% degradation on M4 Pro.**
3. ~~Does the OS page cache help with Zipf expert access?~~ **Yes — 63-78% residency.**
4. Can C/Metal staging buffers achieve true zero-copy? (Phase 4b, #17)
5. What is real tok/s with actual Qwen3-30B-A3B weights (not synthetic)?
6. How does performance scale with longer contexts (larger KV cache eating into headroom)?
7. Can expert prefetch (predicting next-layer routing) hide SSD latency entirely?

## Optimization Roadmap

| Phase | Expected Impact | Status |
|-------|----------------|--------|
| Phase 4a: MLX streaming baseline | ~6-20 tok/s | **Done** |
| Phase 4b: C/Metal zero-copy (#17) | Eliminate MLX copy overhead | Planned |
| Zipf-aware cache warming | Keep hot experts in RAM proactively | Planned |
| Metal dequant shaders (#10) | GPU-side dequant from staging buffer | Planned |
| Expert prefetch (H2) | Hide SSD latency behind compute | Planned |
| Adaptive precision (H1) | Use Q2 for easy tokens, Q4 for hard | Planned |

## Risks

- Real-world expert access may be less Zipf-skewed than assumed → lower cache hit rates
- Long-context inference grows KV cache, reducing available RAM for expert caching
- MLX framework overhead may dominate at small expert sizes — C/Metal path may be required for production speed
- Dequantization on CPU (current path) adds latency vs GPU-side dequant
