# Context: Safetensors Direct Streaming (H8)

**Issue**: #30

## Current State

Complete. Implementation done, 5 code review rounds completed, all high/medium findings addressed. Ready for PR.

## Key Findings

1. **mx.load on safetensors is lazy mmap** — returns in ~0.5ms, actual data read happens at mx.eval. This eliminates the npz double-serialization overhead.
2. **Phase 0: 5.9x speedup** — safetensors p50=21.4ms vs npz p50=126.3ms per block on 72B (single-block, warm cache).
3. **Phase 1: 80 blocks, 26 tensors/block** — 470.9 MB/block Q4, 5 cross-shard blocks out of 80, 29.4 GB total streaming. Cold-cache I/O ceiling: 0.187 tok/s.
4. **Phase 3: Coherent output achieved** — "the self-attention mechanism. Can you explain how" (vs degenerate "the following the following..." before fix).
5. **Root cause of degenerate output**: Missing KV cache. Both predecessor (npz) and initial safetensors implementation passed `cache=None` to every layer, losing all positional context and attention history between tokens. Fixed by creating proper KV cache with `make_prompt_cache(model)`, running prefill on full prompt, then passing cache during decode.
6. **tok/s = 0.007** — block_load_p50=1062ms at 72B scale. The 29.4 GB Q4 working set thrashes the ~9 GB page cache. Cold-cache I/O dominates.
7. **Memory stable**: RSS variance 3 MB, available > 4 GB, pageouts 201 MB.
8. **16 GB projection**: derived from measured 24 GB run values.

## Benchmark Summary

| Metric | Baseline (Phase 3 npz) | Current (safetensors) | Delta |
|--------|------------------------|----------------------|-------|
| tok/s  | 0.005                  | 0.007                | +40%  |
| block load p50 | 704 ms          | 1062 ms              | -51%* |
| peak_rss_mb | 3250              | 5583                 | +72%  |
| coherent output | No            | Yes                  | Fixed |
| reproducible | untested         | Yes                  | New   |

*Block load p50 is worse because predecessor measured swap_ms (CPU→MLX copy) separately from wait_ms (disk I/O). The safetensors p50 includes the full path including page cache misses.

## Review Stats

- Plan review rounds: 8
- Code review rounds: 5
- Total findings addressed: 22+ (plan review) + 13 (code review)

## Code Review Findings Addressed

### Round 1 (3 findings)
- HIGH: Idempotence check now exercises full assign/evict/reload/forward path
- MEDIUM: _find_hf_cache_path uses refs/main for deterministic resolution
- MEDIUM: Reproducibility compares full logits vectors, not just argmax

### Round 2 (3 findings)
- MEDIUM: Fixed 1000x unit error in Phase 1 cold-cache tok/s ceiling
- MEDIUM: NPZ dirs force-recreated at phase start to avoid stale baselines
- MEDIUM: 16 GB projection derived from measured run values

### Round 3 (1 finding addressed, 1 false positive)
- MEDIUM: Clarified 16 GB projection variable naming
- FALSE POSITIVE: mx.get_active_memory() is correct modern API

### Round 4 (2 high/medium + 3 low)
- HIGH: shard_cache scoped per-block to avoid holding refs to all shards
- MEDIUM: Reproducibility gate uses 1e-4 tolerance instead of exact equality
- LOW: Steady-state stats edge case, Phase 1 shard cache, comment wording

### Round 5 (3 medium)
- MEDIUM: Respect HF_HUB_CACHE / HF_HOME env vars
- MEDIUM: evict_block now zeros all params including RMSNorm weights
- MEDIUM: Fixed stale experiments.jsonl cold_tok_s_ceiling (186.891 → 0.187)

## Blockers

- tok/s gate (>=0.1) cannot be met with Q4 streaming blocks — 29.4 GB working set exceeds page cache. Follow-up H8b (Q2 blocks, 16.8 GB working set) required for performance target.

## Next Steps

1. Create PR with H8a mechanism validation results
2. File follow-up issue for H8b (Q2 streaming, 0.5-2 tok/s target)
