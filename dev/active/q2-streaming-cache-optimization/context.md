# H8b: Q2 Streaming with Cache Optimization — Context

**Status**: Implementation complete (Phases 0-2), code reviewed, ready for execution
**Issue**: #32
**Branch**: `experiment/q2-streaming-cache-optimization`

## Current State

- Plan reviewed through 7 automated rounds, approved by user
- Phases 0, 0b, 1, 2 implemented in `scripts/q2_streaming_cache_opt.py`
- Phases 3-5 stubbed out for future implementation
- Code reviewed through 3 automated rounds, all material findings resolved
- Ready to run on hardware with Qwen2.5-72B-Instruct-4bit

## Key Decisions

- Using MLX's native Q2 quantization (bits=2, group_size=64)
- Dequantize Q4 → float16 → re-quantize to Q2 (no FP16 checkpoint available)
- Save Q2 blocks as separate safetensors shards (~4 GB each, 4 shards)
- Mixed-precision: 8+8 Q4 resident blocks (first/last) + 64 streamed Q2 blocks
- Per-block shard scope (not token-scoped) to avoid holding all shards in memory
- Phase 2 NLL is directional self-score only; Phase 4 uses teacher forcing for quality gate
- mincore() via libc.mmap for page cache residency measurement on macOS

## Dependencies

- H8a code: `scripts/safetensors_direct_stream.py` (SafetensorsBlockIndex, load/assign/evict)
- Qwen2.5-72B-Instruct-4bit in HF cache
- MLX 0.31.1+, mlx_lm 0.31.1+

## Implementation Summary

### Phase 0: Q2 Block Micro-benchmark
- Quantizes a single block Q4→FP16→Q2, measures load/assign/forward times
- Computes hit-rate projection: `required_hit_rate = (t_total_p50 - target) / (t_total_p50 - t_forward_p50)`
- Early kill gate if required hit rate > 95%

### Phase 0b: Quality Pilot
- Tests Q2 quality on 7B model with teacher forcing against Q4 reference
- Measures NLL delta (Q2 vs Q4), kills if delta > 3.0

### Phase 1: Q2 Checkpoint Preparation
- Quantizes all 64 streaming blocks to Q2 safetensors shards
- ~4 GB per shard, 4 shards total (~16.8 GB vs 29.4 GB Q4)

### Phase 2: Mixed-precision Streaming
- Full prefill + decode loop with Q4 resident + Q2 streamed blocks
- KV cache persists across block evictions
- Measures tok/s, per-block load times, page cache residency via mincore()

## Findings from Code Review

1. **Shard cache scope** (high): Initial implementation shared shard_cache across layers, causing 13x re-mmap. Fixed to per-block scope, then reverted token-scoped back to per-block to avoid pinning all shards (~16.8 GB).
2. **KV cache missing** (medium): NLL helper initially passed None for KV cache. Fixed with proper `make_prompt_cache(model)`.
3. **NLL self-scoring** (medium): NLL scored against own argmax. Added teacher forcing with `reference_tokens` parameter for Phase 0b/4.
4. **Phase 2 NLL clarification** (low): Documented that Phase 2 self-scored NLL is a degeneration detector, not a quality gate.

## Review Stats

- Plan review rounds: 7
- Code review rounds: 3 (material findings in rounds 1-2, clean in round 3)
- Total findings addressed: 25+ (5 high, 14+ medium, 6+ low)
