# Experiment: Safetensors Direct Streaming (H8)

**Issue**: #30

## Hypothesis

By loading streaming blocks directly from safetensors shards via memory-mapped I/O instead of re-serialized npz files, we can eliminate the serialization overhead and achieve 0.1-0.18 tok/s on Qwen2.5-72B with 24 GB RAM using the original Q4 checkpoint.

**Scope**: This is **H8a — mechanism validation** — proving that the safetensors direct-load path works correctly and delivers a 20x+ improvement over the npz baseline. The issue's 0.5-2 tok/s target requires Q2 streaming blocks with a smaller working set that fits the page cache; that will be **H8b — performance validation** as a follow-up experiment. At completion, issue #30 will be updated to note H8a results and a new issue filed for H8b with the 0.5-2 tok/s target.

**Target hardware**: 24 GB M4 Pro (de-risking precursor). This experiment validates the safetensors direct-load mechanism at 72B scale.

**16 GB promotion gate**: At Phase 3 completion, we will measure and record all deterministic memory components using the project's pinned-memory model: `total_pinned = resident_block_bytes + fixed_non_block_bytes (embeddings + lm_head + norm, ~1.9 GB on 72B) + kv_cache_bytes + metal_scratch + os_overhead`. A follow-up 16 GB experiment is viable if `total_pinned < 10.5 GB` (leaving ~5.5 GB for page cache). Note: 16 GB viability must be confirmed with a separate validation run on target hardware with Q2 streaming blocks — the 24 GB/Q4 measurements are informational only.

## Approach

Replace the npz save/load path (current bottleneck: 704 ms/block) with direct safetensors loading. The key insight is that even if MLX copies mmap'd data internally (confirmed in Phase 1b), the safetensors path eliminates the **double serialization** overhead: np.savez writes to disk, then np.load reads back, then mx.array copies again. With safetensors, we go straight from mmap → mx.load → module assignment — one copy instead of three.

**Note on MLX copy behavior**: Phase 1b confirmed that `mx.array(numpy_data)` copies internally, but also confirmed that GC reclaims old tensors within 2 cycles and RSS stays flat in steady state. The safetensors path doesn't need true zero-copy — it needs to be faster than npz by eliminating the intermediate disk format. If `mx.load` on safetensors is still too slow, the fallback path is C/pread/Metal staging buffers (issue #17).

### Current path (slow)
```
model.safetensors → mx.load() → mx.eval() → np.savez() → disk
disk → np.load() → mx.array() → mx.eval() → forward
Latency: 704 ms/block (471 MB Q4)
```

### Proposed path
```
model.safetensors → mmap'd directly (no intermediate save)
Per-token: selective safetensors load → assign → forward
```

### Theoretical tok/s ceiling (recalculated for actual Q4 block sizes)

Phase 3 streams 64 blocks × 471 MB Q4 = 29.4 GB per token.

**Cold-cache model (default assumption for Q4 blocks)**:
- At 5.5 GB/s cold NVMe: 471 MB / 5.5 GB/s ≈ 86 ms/block
- 64 blocks × 86 ms = 5.5s I/O per token
- Plus mx.eval overhead per block (TBD — Phase 0b measures this component)
- **Expected: ~0.1-0.18 tok/s** (dominated by I/O + eval)

**Why cold-cache is the right default**: The 64-block × 471 MB cyclic sequential scan creates a 29.4 GB working set against a ~9 GB page cache. Under LRU-like eviction, token N+1's early cold misses evict token N's tail blocks before they're reused. Inter-token cache hits are expected to be near-zero. Any warm-cache improvement must be proven empirically (Phase 4), not assumed.

**Target range**: 0.1-0.18 tok/s on 24 GB with Q4 streaming blocks (20-36x improvement over npz baseline). The issue's 0.5-2 tok/s target requires Q2 streaming blocks (262 MB each, smaller working set, better cache fit) — that is a follow-up optimization.

## Implementation Phases

### Phase 0: Shard Layout + MLX Load Characterization (Hard Gate)
**Goal**: (a) Validate that per-block selective safetensors access is feasible, and (b) prove the mx.load safetensors path is faster than npz.

**Phase 0a: Shard layout validation**
1. For Qwen2.5-7B-Instruct-4bit (already downloaded), parse shard headers to determine:
   - Number of shards, shard sizes
   - Per-block tensor count and total bytes
   - Block-to-shard fanout: how many shards does each block's tensors span?
2. Determine the best loading strategy:
   - If `mx.load` supports a `keys` parameter: use it for per-block selective loading (bounded memory)
   - If `mx.load` loads entire shards: check shard sizes. If shards are small enough that loading a full shard is acceptable, proceed. If shards are multi-GB (as expected for 72B), test the `safetensors` Python library for direct byte-offset reads as an alternative
   - Document the chosen strategy and its memory bound

**Phase 0b: Load cycle characterization (with component breakdown)**
1. Using the chosen loading strategy from 0a, run 10 cycles of: load one block's tensors → assign to QuantizedLinear → `mx.eval(block.parameters())` → forward → evict
2. Measure per-cycle with **component-level timing**:
   - `t_load`: time to load/read tensors from safetensors (disk I/O + parsing)
   - `t_assign`: time to assign tensors to QuantizedLinear module attributes
   - `t_eval`: time for `mx.eval(block.parameters())` (Metal materialization)
   - `t_forward`: time for one forward pass through the block
   - `t_evict`: time to evict block (replace with placeholders)
   - `t_total`: end-to-end wall clock
   - RSS delta (should be ~0 after warmup cycle 2)
   - Metal memory delta (via `mx.metal.get_active_memory()`)
3. Same component breakdown for npz equivalent: `np.load` + `mx.array` + `mx.eval` + forward + evict
4. The component breakdown identifies where time is spent — if `t_eval` dominates (as suspected from the 704 ms npz observation), then the safetensors path won't help much and the bottleneck is in MLX's Metal sync, not serialization format

**Gate** (performance-based, not layout-based):
- Per-block loading strategy identified with bounded memory (total bytes materialized per block ≤ 2× block size)
- Per-block end-to-end latency (load + assign + mx.eval) approaching raw I/O floor: must be < 200 ms for a 7B block (~160 MB) — i.e., within 5× of the raw pread latency (41 ms from Phase 1b)
- RSS flat after 2 warmup cycles (max delta < 10 MB in steady state)
- Safetensors path total latency < npz path total latency for the same block
- Record shard fanout as diagnostic info (how many shards per block, total bytes touched), but do NOT abort based on shard count alone
- If no loading strategy achieves < 200 ms per 7B block: abort, file follow-up for C/pread/Metal staging (issue #17)

**Logging**: All Phase 0 results appended to `experiments.jsonl` via `log_experiment()`.

### Phase 1: Safetensors Block Index Builder
**Goal**: Build a mapping from transformer block index → list of (shard_file, tensor_name) tuples.

1. Download `model.safetensors.index.json` for Qwen2.5-72B-Instruct-4bit
2. Parse the weight_map to group tensors by block number (e.g., `model.layers.42.self_attn.q_proj.weight` → block 42)
3. Also capture non-block tensors (embed_tokens, lm_head, norm) separately
4. Validate that all expected tensors per block are present (weight, scales, biases for each QuantizedLinear)
5. Compute per-block byte sizes from shard headers to cross-check against Phase 0 budget numbers (471 MB/block Q4)

**Output**: `SafetensorsBlockIndex` class with:
- `block_tensors(block_idx) → dict[tensor_name → shard_file]`
- `non_block_tensors() → dict[tensor_name → shard_file]`
- Summary statistics printed for validation

**Gate**: Index accounts for all 80 blocks, per-block sizes match Phase 0 budget (±5%)

**Bytes-touched recalculation**: After building the 72B index, compute actual bytes read per block for the chosen loader strategy (accounting for shard fanout, partial shard reads, or full shard loads). Recalculate the cold-cache tok/s ceiling: `64 blocks × actual_bytes_per_block / NVMe_bandwidth`. If revised ceiling < 0.1 tok/s (i.e., actual I/O per block exceeds ~860 ms at 5.5 GB/s), document the finding and evaluate whether the Phase 3 tok/s gate needs adjustment before proceeding.

**Logging**: Index summary and bytes-touched analysis appended to `experiments.jsonl` via `log_experiment()`.

### Phase 2: Direct Load Benchmark on 7B
**Goal**: Prove safetensors direct loading is faster than npz on a small model, with a single-block feasibility gate before scaling to 72B.

**Phase 2a: Single-block feasibility gate**
1. Load Qwen2.5-7B-Instruct-4bit via `mlx_lm.load()`
2. Identify the safetensors shard paths on disk (from HuggingFace cache)
3. Build the block index for 7B
4. For ONE real block, measure the full end-to-end path:
   `mx.load(shard) → filter to block keys → assign to QuantizedLinear modules → mx.eval(block.parameters())`
5. Measure: load latency (ms), RSS delta, Metal memory delta
6. Compare to fresh npz t_total from Phase 0b (same block, same machine, same measurement boundaries — NOT the predecessor's 16.9 ms load-only figure)

**Gate 2a**: Single-block safetensors t_total < npz t_total from Phase 0b (apples-to-apples). If this fails, the mechanism is invalid — stop and investigate before proceeding.

**Phase 2b: Full 7B side-by-side benchmark**
1. Benchmark two paths across all streaming blocks:
   - **npz path** (current): save_block_to_disk → load_block_weights_from_disk → swap_block_weights
   - **safetensors path** (new): mx.load(shard) → filter → assign to block modules
2. For each path, measure:
   - Per-block load latency (ms) — p50/p95
   - RSS before/after (stability)
   - gpu_memory_mb (Metal memory via mx.metal.get_active_memory / get_peak_memory)
   - Logit correctness: forward pass on same input must produce identical logits

**Implementation detail**: For the safetensors path, we need to:
- Use `mx.load(shard_path)` to get a flat dict of all tensors in that shard
- Filter to just the block's tensors
- OR: explore if `mx.load` supports a `keys` parameter for selective loading
- Assign weights directly to the QuantizedLinear modules

**Gate 2b**: safetensors path latency < 50% of npz path latency, logits match exactly

**Logging**: Phase 2a and 2b results appended to `experiments.jsonl` via `log_experiment()`.

### Phase 3: 72B Integration
**Goal**: Run 72B with safetensors direct loading and measure tok/s.

1. Adapt `ssd_lod_72b_integration.py` to use safetensors loading instead of npz
2. Key changes:
   - **Setup**: Skip the npz save step entirely. Just evict streaming blocks after `mx.eval` (no save needed since we load from original safetensors).
   - **Forward pass**: Replace `load_block_weights_from_disk` with safetensors direct load
   - **Weight assignment**: Map safetensors tensor names to QuantizedLinear attributes (weight, scales, biases)
3. Configuration: **16 resident Q4 / 64 streamed Q4** (new name: `16r-Q4 / 64s-Q4`)
   - First 8 + last 8 blocks kept resident in RAM (same split as predecessor Phase 3)
   - 64 streaming blocks loaded directly from original safetensors shards as Q4 (471 MB each)
   - This is NOT the predecessor's "8+8 Q4 / 64 Q2" config — all blocks are Q4 from the original checkpoint
   - Mixed-precision Q2 streaming is a follow-up optimization requiring a separate quantized checkpoint
4. Generate 10+ tokens, measure:
   - tok/s (primary metric)
   - Per-block load latency (ms) — p50/p95
   - peak_rss_mb, available memory (GB)
   - gpu_memory_mb (Metal active/peak memory)
   - **Correctness (72B-specific)**: Since all-resident doesn't fit in 24 GB, use three checks:
     - (a) **Run-to-run reproducibility**: Run the streaming forward pass twice on the same deterministic prompt (same seed, same block order). Compare final logits — must be bitwise identical across runs. This proves the safetensors load path is deterministic and idempotent.
     - (b) **Per-block reload idempotence**: For 3 representative blocks (early ~12, middle ~40, late ~68), load from safetensors, evict, reload again, and verify weights are bitwise identical. This proves the load/assign path doesn't corrupt or drift.
     - (c) **Short-sequence NLL + coherence**: Compute NLL on first 32 tokens of a fixed calibration sequence. Record absolute value for cross-run stability. Generated text must not be degenerate/repetitive (unlike predecessor's "the following the following...").
   - Note: The hard perplexity gate (delta < 0.5 PPL vs all-resident) is tested on 7B in Phase 2b. 72B correctness relies on reproducibility (hard gate), reload idempotence (hard gate), and NLL + coherence (diagnostic).
   - pageins / pageouts (MB)

**Gate**:
- tok/s >= 0.1 (20x improvement over Phase 3's 0.005)
- Run-to-run reproducibility: bitwise identical logits across two streamed runs
- Per-block reload idempotence: weights bitwise identical after evict/reload cycle
- Coherent output (not degenerate)
- Memory stability: available memory > 2 GB throughout generation, pageouts < 500 MB/run, RSS variance < 500 MB across tokens (no unbounded growth)

**16 GB projection (required deliverable)**: Compute a projected 16 GB/Q2 memory budget using measured data from this 24 GB/Q4 run plus the project's memory model:
- `fixed_non_block_bytes`: measured from 24 GB run (embeddings + lm_head + norm, expected ~1.9 GB)
- `resident_block_bytes_q2`: project from Phase 0 budget (262 MB/block × N_resident)
- `kv_cache_bytes`: measured from 24 GB run (scales with context length, not block format)
- `metal_scratch`: measured peak from 24 GB run (expected similar for Q2)
- `os_overhead`: measured from 24 GB run
- `projected_total_pinned = fixed + resident_q2 + kv + scratch + os`
- `projected_page_cache = 16 GB - projected_total_pinned`
- **Pass/fail**: `projected_total_pinned < 10.5 GB` AND `projected_page_cache > 5 GB`
- Note: This is a projection, not a validation. A separate 16 GB run is required to confirm.

**Logging**: All Phase 3 results appended to `experiments.jsonl` via `log_experiment()`.

### Phase 4: Cache Optimization (deferred to follow-up H8b)

Phase 4 cache optimization is **out of scope** for this experiment. The `16r-Q4 / 64s-Q4` config creates a 29.4 GB working set against ~9 GB page cache — near-zero inter-token cache reuse by design. Cache optimization only becomes meaningful with Q2 streaming blocks (16.8 GB working set, ~9 GB cache → ~55% coverage).

**Follow-up experiment (H8b)** will:
1. Build or obtain Q2-quantized safetensors for streaming blocks
2. Run with `8+8 Q4 / 64 Q2` config where cache meaningfully covers the working set
3. Test `madvise(MADV_SEQUENTIAL)`, readahead tuning, and cache residency measurements
4. Target the issue's 0.5-2 tok/s goal

This experiment (H8a) validates the mechanism; H8b validates the performance target.

## Metrics

- [x] tok/s (decode)
- [x] per-block load latency (ms) — p50/p95
- [x] peak_rss_mb
- [x] gpu_memory_mb (Metal active/peak via mx.metal.get_active_memory / get_peak_memory)
- [x] available memory (GB)
- [x] pageins / pageouts (MB)
- [x] logit correctness (exact match vs baseline on 7B; run-to-run reproducibility + reload idempotence on 72B)
- [x] perplexity delta < 0.5 PPL on 7B (all-resident reference)
- [x] short-sequence NLL on 72B (absolute, for stability tracking across runs)
- [ ] cache_hit_rate (Phase 4 only)

## Baseline

Phase 3 npz results:
- tok/s: 0.005 (188s/tok)
- Block swap p50: 704 ms
- Block wait p50: 1111 ms

Phase 2b raw I/O (theoretical floor):
- 262 MB block read: 55 ms p50
- 0.27 tok/s (64 blocks, all cold)

```bash
# Phase 2 benchmark (7B side-by-side)
uv run python scripts/safetensors_direct_stream.py --phase 2 --model mlx-community/Qwen2.5-7B-Instruct-4bit

# Phase 3 benchmark (72B integration)
uv run python scripts/safetensors_direct_stream.py --phase 3 --model mlx-community/Qwen2.5-72B-Instruct-4bit --n-tokens 10
```

## Success Criteria

1. **Phase 0**: Per-block selective loading proven with bounded memory, end-to-end latency < 200 ms/block on 7B, component breakdown recorded, RSS flat after warmup, safetensors total latency < npz total latency (hard gate — abort if no loading path achieves < 200 ms)
2. **Phase 2a**: Single-block safetensors load+assign+eval latency < npz equivalent (feasibility gate)
3. **Phase 2b**: safetensors load latency < 50% of npz latency on 7B, logits match exactly, perplexity delta < 0.5
4. **Phase 3**: tok/s >= 0.1 on 72B (20x improvement over 0.005 baseline), run-to-run reproducible, reload idempotent, coherent output, memory stable (avail > 2 GB, pageouts < 500 MB), 16 GB projection computed
5. **Phase 4**: Deferred to follow-up H8b (Q2 streaming blocks required for meaningful cache optimization)

Overall success (H8a mechanism validation): tok/s >= 0.1 on 72B with stable memory and coherent output (20x improvement over npz baseline)

## Risks

1. **MLX mx.load copies mmap'd data**: Even with safetensors, MLX may internally copy data rather than use the mmap view. Phase 1b found that mx.array(numpy_data) causes a copy — but mx.load's safetensors path may be different.
2. **Shard boundary misalignment**: A single block's tensors may span multiple shards, requiring multi-file loading per block.
3. **QuantizedLinear metadata**: scales/biases may not be stored as separate tensors in safetensors — need to verify naming convention.
4. **Output quality**: Phase 3's degenerate text ("the following the following...") may not be a serialization bug — could be a weight restoration logic issue that persists with safetensors path.

## Rollback

Revert to Phase 3 npz-based implementation (`ssd_lod_72b_integration.py`). The npz path is slow but mechanically correct.
