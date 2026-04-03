# H8b: Q2 Streaming with Cache Optimization — Context

**Status**: In progress — plan drafted, awaiting review
**Issue**: #32
**Branch**: `experiment/q2-streaming-cache-optimization`

## Current State

- Plan written in `plan.md`
- Branch created from master
- Pending: plan review, then implementation

## Key Decisions

- Using MLX's native Q2 quantization (bits=2, group_size=64)
- Dequantize Q4 → float16 → re-quantize to Q2 (no FP16 checkpoint available)
- Save Q2 blocks as separate safetensors shards (~4 GB each, 4 shards)
- Mixed-precision: 16 resident Q4 blocks + 64 streamed Q2 blocks

## Dependencies

- H8a code: `scripts/safetensors_direct_stream.py` (SafetensorsBlockIndex, load/assign/evict)
- Qwen2.5-72B-Instruct-4bit in HF cache
- MLX 0.31.1+, mlx_lm 0.31.1+

## Findings

(To be updated as phases complete)

## Review Stats

- Plan review rounds: 7
- Code review rounds: 0
- Findings addressed: 20+ (4 high, 12+ medium, 4+ low)
