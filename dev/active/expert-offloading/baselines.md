# Baseline Comparison: Standard vs Expert Offloading

**Hardware**: M4 Pro, 24GB unified memory, NVMe SSD (15-17 GB/s warm)

## Standard MLX (everything in RAM)

| Model | Params | Size @ 4-bit | Fits 24GB? | Fits 16GB? | Speed (est.) |
|-------|--------|-------------|------------|------------|-------------|
| Dense 7B (e.g. Llama 3.1 7B) | 7B | ~4 GB | Yes | Yes | ~100+ tok/s |
| Dense 14B (e.g. Qwen2.5-14B) | 14B | ~8 GB | Yes | Yes | ~40-50 tok/s |
| Qwen3-30B-A3B (MoE) | 30B (3B active) | 14.0 GB | Tight | No | ~60-80 tok/s |
| Mixtral-8x7B (MoE) | 46.7B (12.9B active) | 21.8 GB | No | No | N/A |

Notes:
- "Tight" = model loads but leaves little room for KV cache and OS overhead
- Rule of thumb: model should be <60-70% of total RAM for comfortable inference
- Qwen3-30B-A3B @ 4-bit on 24GB leaves ~10GB for OS + KV, workable for short contexts
- On 16GB, the largest practical dense model is ~7-8B

## With Expert Offloading (Phase 4a results)

Only non-expert weights + active experts need to be in RAM. Expert weights stream from SSD.

| Model | RAM needed | On SSD | Fits 24GB? | Fits 16GB? | Speed (est.) |
|-------|-----------|--------|------------|------------|-------------|
| Qwen3-30B-A3B @ 4-bit | ~2 GB | 13.5 GB | Easily | **Yes** | ~6-20 tok/s |
| Qwen3-30B-A3B @ 3-bit | ~1.5 GB | 10.1 GB | Easily | **Yes** | ~8-25 tok/s |
| Mixtral-8x7B @ 4-bit | ~6.5 GB | 21 GB | Yes | **Yes** | ~3-10 tok/s |
| Mixtral-8x7B @ 3-bit | ~5.0 GB | 15.8 GB | Yes | **Yes** | ~4-12 tok/s |

### RAM breakdown (Qwen3-30B-A3B @ 4-bit)

| Component | Size |
|-----------|------|
| Non-expert weights | 0.53 GB |
| Active experts in flight (384 x 2.25 MB) | 0.84 GB |
| KV cache (short context) | ~0.5 GB |
| **Total** | **~1.9 GB** |

### Speed estimate methodology

- Phase 4a measured p50 = 0.40 ms per expert call (streamed from SSD)
- Qwen3-30B-A3B: 384 expert calls per token (8 active x 48 layers)
- Floor: 384 x 0.40 ms = 154 ms/token = ~6.5 tok/s (all experts cold)
- With page cache (63-78% Zipf residency), hot experts are near-instant
- Estimated practical range: 6-20 tok/s depending on cache warmth

## Summary: What changes

| Metric | Standard (16GB) | With Offloading (16GB) | Improvement |
|--------|-----------------|----------------------|-------------|
| Max model size | ~7-8B dense | **30B MoE** | ~4x |
| Best quality achievable | 7B-level | **30B-level (3B active)** | Major |
| Speed at max size | ~100 tok/s (7B) | ~6-20 tok/s (30B) | Slower |
| Speed/quality tradeoff | Limited by RAM | Limited by SSD bandwidth | Better quality per tok/s |

| Metric | Standard (24GB) | With Offloading (24GB) | Improvement |
|--------|-----------------|----------------------|-------------|
| Max model size | 30B MoE (tight) | **46B+ MoE comfortably** | ~1.5x+ |
| Headroom for KV cache | Minimal at 30B | ~22 GB free at 30B | Massive |
| Long context support | Limited | **Full** | Major |

## Key insight

The trade-off is speed for quality. A 30B MoE at 10 tok/s produces significantly better outputs than a 7B dense model at 100 tok/s. Expert offloading makes this possible on hardware that otherwise can't run large models at all.

## Optimization roadmap

| Phase | Expected impact | Status |
|-------|----------------|--------|
| Phase 4a: MLX streaming (current) | Baseline ~6-20 tok/s | Done |
| Phase 4b: C/Metal zero-copy (#17) | Eliminate MLX copy overhead | Planned |
| Zipf-aware cache warming | Keep hot experts in RAM | Planned |
| Metal dequant shaders (#10) | GPU-side dequant, no CPU copy | Planned |
| Prefetch next-layer experts | Hide SSD latency behind compute | Planned |

Each optimization should push the speed floor higher while maintaining the RAM savings.
