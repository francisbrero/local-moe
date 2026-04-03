# Context: Safetensors Direct Streaming (H8)

**Issue**: #30

## Current State

Implementation complete. Phase 0-3 all ran. Degenerate output bug found and fixed (missing KV cache). Coherent output confirmed. tok/s gate (>=0.1) fails due to page cache thrashing at Q4 scale — expected per plan (H8b needed for performance target).

## Key Findings

1. **mx.load on safetensors is lazy mmap** — returns in ~0.5ms, actual data read happens at mx.eval. This eliminates the npz double-serialization overhead.
2. **Phase 0: 5.9x speedup** — safetensors p50=21.4ms vs npz p50=126.3ms per block on 72B (single-block, warm cache).
3. **Phase 1: 80 blocks, 26 tensors/block** — 470.9 MB/block Q4, 5 cross-shard blocks out of 80, 29.4 GB total streaming.
4. **Phase 3: Coherent output achieved** — "the self-attention mechanism. Can you explain how" (vs degenerate "the following the following..." before fix).
5. **Root cause of degenerate output**: Missing KV cache. Both predecessor (npz) and initial safetensors implementation passed `cache=None` to every layer, losing all positional context and attention history between tokens. Fixed by creating proper KV cache with `make_prompt_cache(model)`, running prefill on full prompt, then passing cache during decode.
6. **tok/s = 0.007** — block_load_p50=1062ms at 72B scale. The 29.4 GB Q4 working set thrashes the ~9 GB page cache. Cold-cache I/O dominates.
7. **Memory stable**: RSS variance 3 MB, available > 4 GB, pageouts 201 MB.
8. **16 GB projection viable**: projected_total=9.0 GB, page_cache=7.0 GB.

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
- Code review rounds: 0
- Total findings addressed: 22+ (plan review)

## Blockers

- tok/s gate (>=0.1) cannot be met with Q4 streaming blocks — 29.4 GB working set exceeds page cache. Follow-up H8b (Q2 blocks, 16.8 GB working set) required for performance target.

## Next Steps

1. Code review loop
2. Finalize and create PR with H8a mechanism validation results
3. File follow-up issue for H8b (Q2 streaming, 0.5-2 tok/s target)
