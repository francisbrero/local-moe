# H8b: Q2 Streaming with Cache Optimization — Context

**Status**: All phases implemented (0-5), code reviewed, ready for execution
**Issue**: #32
**Branch**: `experiment/q2-streaming-cache-optimization`

## Current State

- Plan reviewed through 7 automated rounds, approved by user
- All phases (0, 0b, 1, 2, 3, 4, 5) implemented in `scripts/q2_streaming_cache_opt.py`
- Code reviewed through 9 automated rounds, all material findings resolved
- Ready to run on hardware with Qwen2.5-72B-Instruct-4bit

## Key Decisions

- Using MLX's native Q2 quantization (bits=2, group_size=64)
- Dequantize Q4 → float16 → re-quantize to Q2 (no FP16 checkpoint available)
- Save Q2 blocks as separate safetensors shards (~4 GB each, 4 shards)
- Mixed-precision: 8+8 Q4 resident blocks (first/last) + 64 streamed Q2 blocks
- Per-block shard scope (not token-scoped) to avoid holding all shards in memory
- Phase 2 NLL is directional self-score only; Phase 4 uses teacher forcing for quality gate
- mincore() via libc.mmap for page cache residency measurement on macOS
- Q2 cache dir namespaced by model ID to prevent stale shard corruption
- OS page cache eviction via madvise(MADV_DONTNEED) between A/B configs

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

### Phase 3: Cache Optimization
- Synthetic madvise benchmark to validate if advisory helps on Apple Silicon
- A/B comparison: none vs prefault vs readahead (with OS page cache eviction between configs)
- Per-block residency measurement via block_byte_range() + mincore()
- Prefault: sidecar mmap with page-aligned offsets to pre-warm block pages
- Readahead: background thread pread() targeting next block's actual byte range

### Phase 4: Quality Validation
- Phase 4a: 7B PPL comparison (all-Q4 vs mixed Q4/Q2) + 50-token NLL stability
- Phase 4b: 72B teacher-forced NLL comparison (Q2 vs Q4 reference)
- Proper token alignment: score against q4_ref_tokens[tok_i+1] after feeding tok_i

### Phase 5: 16 GB Provisional Projection
- Two configs: 3-Q4/77-Q2 and all-80-Q2 (with Q4 fallback for missing blocks)
- Memory pressure via ballast allocation with verification
- Full model cleanup between configs (blocks, inner, kv_cache, q2_index)
- App budget constraint check (10.5 GB limit)

## Findings from Code Review

### Rounds 1-3 (Phases 0-2)
1. **Shard cache scope** (high): Reverted to per-block scope to avoid pinning all shards
2. **KV cache missing** (medium): Added proper make_prompt_cache(model)
3. **NLL self-scoring** (medium): Added teacher forcing with reference_tokens
4. **Phase 2 NLL clarification** (low): Documented as degeneration detector

### Round 4 (Phases 3-5 initial)
5. **assign_block_weights missing block_idx** (high): Fixed 5 call sites
6. **Teacher forcing seed** (high): Seed with Q4 reference token, not Q2 argmax
7. **7B memory cleanup** (medium): Delete blocks_7b/inner_7b before 72B load
8. **all-80-Q2 label** (medium): Track actual_label when falling back to Q4
9. **Shard-wide prefault** (medium): Prefault only block byte range via header offsets

### Round 5
10. **NLL off-by-one** (high): Score q4_ref_tokens[tok_i+1] not [tok_i]
11. **Coherence check on teacher-forced text** (medium): Removed (always Q4 tokens)
12. **Page-aligned mmap offset** (medium): Align in _prefault_shard_range
13. **Block-range residency** (medium): Use block_byte_range() not whole shard
14. **Phase 5 Q4 fallback metadata** (medium): Add restore_q4_block_metadata

### Round 6
15. **Hoisted Q4 index** (high): Moved out of Phase 5 inner loops
16. **Page-aligned residency** (medium): Align offset in measure_file_residency
17. **Readahead thread joining** (medium): Track and join before next load
18. **Unused kv_cache param** (low): Removed from _run_streaming_pass

### Round 7
19. **A/B fairness** (high): Re-evict blocks between Phase 3 configs
20. **Readahead wrong bytes** (medium): pread block byte range, not first 64KB
21. **Model-scoped cache** (medium): Namespace Q2 cache by model ID
22. **Phase 5 decode Q4 index** (medium): Reuse hoisted q4_index

### Round 8
23. **OS page cache eviction** (medium): madvise(MADV_DONTNEED) between configs
24. **Synthetic bench page eviction** (medium): Evict between madvise samples
25. **Phase 5 inter-config cleanup** (medium): Delete all model refs between configs
26. **Thread join after last block** (medium): Join readahead after inner loop

### Round 9
27. **fd/mmap leak guard** (medium): try/finally in _synthetic_madvise_bench
28. **ballast_gb logging** (low): Zero in 'no ballast needed' branch

## Review Stats

- Plan review rounds: 7
- Code review rounds: 9 (material findings in rounds 1-2, 4-9; clean convergence at round 9)
- Total findings addressed: 28+ (6 high, 18+ medium, 4+ low)
