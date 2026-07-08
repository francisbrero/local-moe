# H8b: Q2 Streaming with Cache Optimization — Performance Validation

**Issue**: #32
**Branch**: `experiment/q2-streaming-cache-optimization`
**Predecessor**: H8a (safetensors direct streaming, PR #31)

## Goal

Achieve 0.5-2 tok/s on Qwen2.5-72B by:
1. Quantizing streaming blocks from Q4 (471 MB) to Q2 (262 MB), shrinking the cyclic working set from 29.4 GB to 16.8 GB
2. Applying cache optimization (madvise, readahead) to improve page cache hit rates
3. Validating that Q2 quality is acceptable (PPL delta < 1.0 on streaming blocks)

## Issue-Level Exit Criteria

**Success**: Issue #32 closes as resolved if tok/s >= 0.5 on the 8+8 Q4/64 Q2 config (or equivalent) with coherent output and PPL delta < 1.0.

**Partial success / negative result**: If tok/s < 0.5 but > 0.1, document findings, keep issue open, and re-scope to either:
- Alternative quantization (Q3, mixed Q2/Q3)
- C/Metal staging buffers (issue #17) for lower-level loader control
- Accept as feasibility study with measured ceiling

**Failure**: If tok/s ≈ cold-cache ceiling (no cache reuse) and < 0.1, document as negative result. The cyclic sequential scan defeats page cache reuse at this working set size. Close issue with findings, file follow-up for alternative approaches.

## Architecture

Builds on H8a's validated infrastructure:
- `SafetensorsBlockIndex` — maps block indices to safetensors shard/tensor pairs
- `load_block_from_safetensors` / `assign_block_weights` / `evict_block` — streaming block lifecycle
- KV cache via `make_prompt_cache` — persists across block evictions
- 8+8 resident (Q4) / 64 streamed config

### Key change: Mixed-precision Q4/Q2

```
Resident (first 8 + last 8):  Q4  — loaded once, stay in memory
Streaming (middle 64):         Q2  — loaded per-token from safetensors via mmap
```

The Q2 blocks will be a separate set of safetensors files produced by re-quantizing the streaming layers. The `SafetensorsBlockIndex` will be pointed at these Q2 files for streaming blocks while resident blocks continue using the original Q4 checkpoint.

## Phases

### Phase 0: Q2 Block Micro-Benchmark (Pre-Gate)

**Script**: `scripts/q2_streaming_cache_opt.py --phase 0`

Before committing to full 72B integration, measure the component-level latency of a single Q2 block to project a realistic tok/s ceiling.

1. Quantize a single representative block (block 40, middle of model) from Q4 to Q2
2. Measure the full load cycle with component breakdown:
   - `t_load_ms` — mx.load of Q2 shard (lazy mmap)
   - `t_assign_ms` — assign Q2 tensors to QuantizedLinear module (with bits/group_size update)
   - `t_eval_ms` — mx.eval to materialize Q2 weights
   - `t_forward_ms` — forward pass through the block
   - `t_evict_ms` — evict block weights
3. Run 10 cycles, report p50 for each component
4. Project per-token ceiling: `64 × t_total_p50` (all cold) and `64 × (cache_hit_rate × t_forward + (1-cache_hit_rate) × t_total)` (estimated warm)
5. Compare Q2 block size to Q4: verify ≈ 56% reduction (262 MB vs 471 MB)
6. **Multi-token residency probe**: After running the 10 load/evict cycles on the single block, simulate a 2-token full scan:
   - Load all 64 streaming blocks sequentially (token 1 scan), measuring `mincore()` residency after each block
   - Immediately re-scan all 64 blocks (token 2 scan), measuring `resident_before_load_fraction` for each
   - Compute `measured_inter_token_hit_rate` = mean of token-2's `resident_before_load_fraction`
   - This directly measures whether macOS LRU evicts early blocks before the second scan reaches them
   - If `measured_inter_token_hit_rate < 0.1`, cache reuse is near-zero and the warm-cache tok/s target must be set to the cold-cache ceiling

**Gate**: Three-tier gate:
- **Hard gate**: `t_total_p50 < 150 ms` — cold-cache ceiling ≥ 0.10 tok/s (minimum viable)
- **Projection gate**: Compute the cache hit rate needed for Phase 2's warm-cache target of 0.3 tok/s:
  - Per-block target latency: `target_per_block = 1000 / (0.3 × 64)` ≈ 52 ms
  - A cache hit reduces per-block cost from `t_total_p50` (cold) to `t_forward_p50` (warm, no I/O)
  - Solving: `hit_rate × t_forward + (1 - hit_rate) × t_total = target_per_block`
  - `required_hit_rate = (t_total_p50 - target_per_block) / (t_total_p50 - t_forward_p50)`, clamped to [0, 1]
  - If `required_hit_rate > 0.80` (or > 1.0), the 0.3 tok/s target is implausible given ~53% cache coverage, and Phase 2's target should be revised downward before proceeding.
  - Log both `target_per_block_ms` and `required_hit_rate` so the gate is auditable.
- **Reuse gate**: `measured_inter_token_hit_rate > 0.1` — confirms that at least some inter-token cache reuse exists. If this fails, the warm-cache hypothesis is invalidated for this working-set size, and Phase 2's tok/s target is set to the cold-cache ceiling. (This doesn't block Phase 2 — we still want to measure end-to-end tok/s — but it calibrates expectations.)

**Metrics logged**: `t_load_ms`, `t_assign_ms`, `t_eval_ms`, `t_forward_ms`, `t_evict_ms`, `t_total_ms`, `rss_delta_mb`, `metal_active_mb`, `metal_peak_mb`, `q2_block_size_mb`, `cold_ceiling_tok_s`, `required_hit_rate_for_0.3`, `measured_inter_token_hit_rate`

### Phase 0b: Q2 Quality Pilot (Early Kill Switch)

**Script**: `scripts/q2_streaming_cache_opt.py --phase 0b`

Before preparing all 64 Q2 blocks, validate that double-quantized Q2 (Q4→FP16→Q2) produces acceptable quality on a small sample.

1. On the 7B model (fits in RAM): dequantize 3 representative blocks (early, middle, late) from Q4→FP16→Q2
2. Run a short NLL comparison (100 tokens on a fixed prompt) between all-Q4 and 3-block-Q2 configs
3. Compare per-token NLL: if NLL delta > 1.0 on even 3 blocks, Q2 may be too aggressive for 64 blocks

**Gate**: Per-token NLL delta < 1.0 on 7B with 3 Q2 blocks. If this fails, pivot to Q3 (3-bit) before investing in full checkpoint preparation.

### Phase 1: Q2 Checkpoint Preparation

**Script**: `scripts/q2_streaming_cache_opt.py --phase 1`

**HARD PREREQUISITE**: Phase 0b quality pilot must pass before this phase runs. If Phase 0b fails (NLL delta >= 1.0), do NOT proceed to Phase 1. Instead, pivot to Q3 (3-bit) quantization and re-run Phase 0/0b with `bits=3`.

1. Load Qwen2.5-72B-Instruct-4bit via mlx_lm (lazy mmap)
2. For each of the 64 streaming blocks (indices 8-71):
   - Materialize the Q4 block weights
   - Dequantize each QuantizedLinear layer (weight × scales + biases → float16)
   - Re-quantize to Q2 with group_size=64 using `mx.quantize(w, group_size=64, bits=2)`
   - Save the Q2 tensors (weight, scales, biases) to a new safetensors file
3. Build a Q2 `SafetensorsBlockIndex` pointing at the new Q2 shards
4. Verify: per-block size ≈ 262 MB, tensor names follow existing pattern

**Gate**: Q2 blocks load correctly, forward pass produces valid (non-NaN) logits on a single block.

**Memory strategy**: Process one block at a time — dequantize, re-quantize, save, evict. Peak memory = 1 dequantized block (float16 ≈ 942 MB) + Q2 output (262 MB) ≈ 1.2 GB additional.

### Phase 2: Mixed-Precision Streaming Integration

**Script**: `scripts/q2_streaming_cache_opt.py --phase 2`

1. Load model with Q4 weights (standard mlx_lm load)
2. Resident blocks (0-7, 72-79): keep at Q4 in memory
3. Streaming blocks (8-71): load from Q2 safetensors per token
4. Reuse H8a's forward pass pattern with KV cache
5. Generate 20 tokens, measure tok/s and block load latency

**Key difference from H8a**: `load_block_from_safetensors` now uses the Q2 index for streaming blocks. The `assign_block_weights` function should work without changes since it assigns tensors by name regardless of quantization level — but we need to handle the case where the Q2 QuantizedLinear has different `bits`/`group_size` than the model's default Q4.

**Assignment approach**: Instead of assigning tensors to the existing Q4 QuantizedLinear modules (which expect Q4 shapes), we'll need to either:
- (a) Replace the module's `bits` and `group_size` attributes before assignment, or
- (b) Call `module.to_quantized(bits=2, group_size=64)` to restructure the module first

We'll test option (a) first as it's simpler.

**Gate**: tok/s >= 0.3 (warm-cache target — requires inter-token page cache reuse; cold-cache ceiling from Phase 0 is the floor), coherent output, memory stable (RSS variance < 500 MB). If Phase 0's projection gate showed the required hit rate is > 80%, this target will be revised downward to the Phase 0 cold-cache ceiling.

### Phase 3: Cache Optimization

**Script**: `scripts/q2_streaming_cache_opt.py --phase 3`

**Primary metric**: End-to-end tok/s A/B comparison (with vs without optimization). This is the only reliable metric since mx.load owns the mmap internally and we cannot directly measure per-run cache hit rates.

**Secondary signal**: `vm_stat` pagein/pageout deltas between tokens — system-wide pressure indicator, not a precise cache hit rate. Logged for directional insight only.

**Approach**:

**Step 1: Synthetic page-cache harness (prerequisite)**
Before wiring any optimization into the real loader, validate that madvise/readahead actually improve sequential read performance on this hardware:
- Create a synthetic benchmark: read a ~3 GB file sequentially with and without `madvise(MADV_SEQUENTIAL)`
- Measure read throughput (GB/s) and `mincore()` residency before/after
- If madvise shows no improvement on macOS/Apple Silicon NVMe, skip Optimization A entirely and accept Phase 2 tok/s as the ceiling

**Step 2: A/B comparison on real workload** (only if Step 1 shows benefit):
1. **Baseline**: Run Phase 2 config as-is (20 tokens), record tok/s and per-token latency distribution
2. **Optimization A — page pre-touch via sidecar mmap**: Before each block load, use a Python-owned read-only mmap of the same shard file to pre-fault the block's page range into the page cache (simple sequential read of the byte range). This explicitly pre-populates the pages that mx.load's mmap will subsequently access. The intervention being tested is "does pre-faulting pages before mx.eval eliminates page-fault latency from the critical path?" A null result means pre-faulting doesn't help (pages were already resident or faulting is not the bottleneck).
3. **Optimization B — readahead priming**: Before each token's forward pass, issue `os.pread()` calls on the *next* token's shard byte ranges in a background thread, overlapping I/O with the current token's compute. This tests whether lookahead prefetching can hide I/O latency.
4. **A/B comparison**: Run each config for 20 tokens, compare steady-state tok/s

**Per-block page residency measurement** (for all configs):

Instrument block-level residency using `mincore()` on the actual byte ranges each streamed block touches. For each block load, measure *before* loading:

```python
import mmap, ctypes, os

# Read-only mmap of same file (doesn't conflict with mx.load's mmap)
# Used ONLY for mincore() measurement, not for data loading
fd = os.open(shard_path, os.O_RDONLY)
mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
page_size = os.sysconf("SC_PAGE_SIZE")

def measure_block_residency(mm, block_offset, block_size):
    """Measure page cache residency for a specific block's byte range."""
    n_pages = (block_size + page_size - 1) // page_size
    vec = (ctypes.c_char * n_pages)()
    addr = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(mm)) + block_offset)
    libc.mincore(addr, ctypes.c_size_t(block_size), vec)
    resident = sum(1 for i in range(n_pages) if vec[i] != b'\x00')
    return resident / n_pages  # fraction [0, 1]
```

Per-token metrics logged:
- `resident_before_load_fraction` — fraction of block's pages already in cache before mx.load
- `cold_bytes_mb` — `(1 - resident_fraction) × block_size_mb` (bytes that must be faulted in)
- `estimated_block_miss_rate` — `1 - mean(resident_before_load_fraction)` across all 64 blocks per token

These metrics directly tie tok/s changes to block-level cache reuse, not just system-wide signals.

**Note**: If neither optimization improves tok/s, we'll document that finding as a negative result. macOS may already optimize NVMe sequential reads sufficiently, making explicit advisory unnecessary. If Python-level fd hints cannot reach MLX's internal mmap path, the correct fallback for lower-level loader control is C/Metal staging buffers (issue #17).

**Gate**: tok/s measurably improved vs Phase 2 baseline (>= 10% improvement). The absolute 0.5 tok/s target is aspirational — if Phase 2 achieves a lower ceiling, Phase 3's gate scales with it.

### Phase 4: Quality Validation

**Script**: `scripts/q2_streaming_cache_opt.py --phase 4`

**Phase 4a: 7B PPL comparison** (controlled environment — both configs fit in RAM):
1. Measure perplexity on a calibration set (WikiText-2 or similar small corpus)
2. Compare two configs on 7B:
   - All-Q4 (reference baseline)
   - 8+8-Q4 / streaming-Q2 (our config, proportionally scaled)
3. Measure NLL stability over sustained 50+ token generation
4. Check for quality-sensitive layers: if PPL is bad, try Q3 for attention layers

**Phase 4b: 72B quality sanity check** (streaming environment):
1. Run 72B Q4-streaming baseline (H8a Phase 3 config): generate 20 tokens with fixed prompt, capture per-token NLL
2. Run 72B Q2-streaming (Phase 2 config): same prompt, capture per-token NLL
3. Compare logit distributions on first 5 generated tokens (KL divergence or max logit diff)
4. Fixed-prompt coherence check: verify Q2 output is coherent and contextually appropriate

**Phase 4c: 16 GB candidate layout quality check** (if Phase 5 is attempted):
1. For each 16 GB candidate layout (3-Q4/77-Q2, all-80-Q2):
   - On 7B (proportionally scaled): compare NLL against all-Q4 baseline using same fixed prompt
   - On 72B (streaming): generate 10 tokens with fixed prompt, check for coherence and NLL delta
2. This ensures no 16 GB layout is declared memory-feasible without direct quality evidence

**Gate**:
- Phase 4a: PPL delta < 1.0 vs Q4 reference on 7B model
- Phase 4b: Per-token NLL delta < 0.5 on 72B, coherent output on fixed prompt (NLL threshold replaces subjective "coherent" check)
- Phase 4c: NLL delta < 1.0 for each 16 GB candidate layout tested

### Phase 5: 16 GB Provisional Projection (Stretch Goal)

**Script**: `scripts/q2_streaming_cache_opt.py --phase 5`

**Scope note**: The primary experiment target is 24 GB with 8+8 Q4/64 Q2. Phase 5 is a stretch goal — a provisional feasibility probe for 16 GB. The main experiment path (Phases 0-4) validates the Q2 streaming mechanism and cache optimization on the dev machine. Phase 5 extends those findings to a 16 GB projection but does not replace actual hardware validation.

**Note**: This phase provides a *provisional* projection only. Results from ballast-based memory pressure on the 24 GB dev machine do not prove behavior on actual 16 GB M4 hardware (different unified-memory contention, OS reclaim behavior, thermal profiles). A confirmation run on real 16 GB hardware is required before making any "fits on 16 GB" claim.

1. Compute actual memory budget from Phase 2 measurements:
   - `measured_os_overhead` = 24 - `available_gb_idle` (from Phase 2 start)
   - `model_working_set` = embeddings + lm_head + norm + KV cache + resident blocks + Metal scratch + transient peak (from Phase 0)
   - **16 GB app budget constraint**: CLAUDE.md states ~10-11 GB usable on 16 GB M4.
     Enforce: `model_working_set + transient_peak <= 10.5 GB`
     If this fails, reduce resident block count until it fits.
   - `page_cache` = 16 - `measured_os_overhead` - `model_working_set`
   - If `page_cache < 2 GB`, flag the config as "memory-feasibility probe only" rather than claiming viability.
2. Project to 16 GB: select config that satisfies the 10.5 GB app budget constraint
3. **Calibrated memory pressure simulation**:
   a. Record `available_gb_idle` before model load
   b. Load model + setup resident blocks, record `available_gb_post_setup`
   c. Compute required ballast: `ballast_gb = available_gb_post_setup - (16 - (24 - available_gb_post_setup))`
      (target: same consumed memory relative to a 16 GB total as measured on 24 GB)
   d. If `ballast_gb <= 0`: available memory is already at or below the 16 GB equivalent — log this and skip ballast
   e. Allocate ballast via `create_memory_pressure()`, then record `available_gb_after_ballast`
   f. **Verify ballast applied**: assert `available_gb_after_ballast < available_gb_post_setup - 0.5 * ballast_gb`
      If verification fails, log a warning and mark the simulation as unreliable
4. Run Phase 2 config under memory pressure, recording all metrics

**16 GB configs to test**:
- 3 Q4 resident (first 2 + last 1) + 77 Q2 streamed
- All 80 Q2 streamed (simplest, no resident blocks)

**Provisional indicators** (informational, NOT acceptance criteria — all are non-blocking):
- No OOM on simulated 16 GB
- tok/s >= 0.1
- `min_available_gb > 1.0` (headroom for OS and transient spikes)
- `pageout_delta_mb < 1000` (not swapping excessively)
- `metal_peak_mb` within measured headroom from Phase 2
- Ballast verification passed

**Acceptance gate** (blocking): No "fits on 16 GB" or throughput claim until the same config is run on actual 16 GB M4 hardware. Phase 5 on the 24 GB dev machine is a feasibility probe only. Results will be logged as `provisional: true` in experiments.jsonl.

## Memory Budget

### Memory reconciliation formula

All budget tables follow this identity:
```
total_ram = os_overhead + app_pinned + transient_peak + free_for_page_cache
```
Where:
- `os_overhead` = `total_ram - available_gb_idle` (measured before model load)
- `app_pinned` = embeddings + lm_head + norm + resident blocks + KV cache + Metal scratch
- `transient_peak` = per-block load/eval spike (measured in Phase 0)
- `free_for_page_cache` = `total_ram - os_overhead - app_pinned - transient_peak`

### Phase 2 (24 GB / 8+8 Q4 / 64 Q2) — nominal estimates, refined by Phase 2 measurements
```
total_ram = 24.0 GB
os_overhead (measured)          =  5.0 GB  (24 - available_gb_idle, nominal)
────────────────────────────────────────────────
App pinned:
  Embeddings + lm_head          =  1.9 GB
  16 resident Q4 blocks         =  7.5 GB
  KV cache (2048 ctx)           =  0.16 GB
  Metal scratch                 =  0.5 GB
  ────────────────────────────
  app_pinned subtotal           = 10.1 GB
Transient peak (1 block)        =  0.3 GB
────────────────────────────────────────────────
free_for_page_cache             =  8.6 GB  (24.0 - 5.0 - 10.1 - 0.3)
Streaming working set           = 16.8 GB  (64 × 262 MB)
Effective cache coverage        = ~51%     (8.6 / 16.8)
```

### Phase 5 (16 GB / 3 Q4 / 77 Q2) — nominal estimates, actual budget derived from measured values
```
total_ram = 16.0 GB
os_overhead (measured)          =  5.0 GB  (same as 24 GB, nominal)
────────────────────────────────────────────────
App pinned:
  Embeddings + lm_head          =  1.9 GB
  3 resident Q4 blocks          =  1.4 GB
  KV cache (2048 ctx)           =  0.16 GB
  Metal scratch                 =  0.5 GB
  ────────────────────────────
  app_pinned subtotal           =  4.0 GB
Transient peak (1 block)        =  0.3 GB
────────────────────────────────────────────────
free_for_page_cache             =  6.7 GB  (16.0 - 5.0 - 4.0 - 0.3)
Streaming working set           = 20.2 GB  (77 × 262 MB)
Effective cache coverage        = ~33%     (6.7 / 20.2)

App budget check (CLAUDE.md: ~10-11 GB usable on 16 GB):
  app_pinned + transient_peak   =  4.3 GB  ✓ (well within 10.5 GB limit)
```

**Note**: `os_overhead` is derived from `total_ram - available_gb_idle` as measured before model load. The 5.0 GB value is nominal — actual measurements replace it. All app component sizes are refined by Phase 2 measurements.

## Success Criteria

| Metric | Phase 0 | Phase 0b | Phase 2 | Phase 3 | Phase 4a-c | Phase 5* |
|--------|---------|----------|---------|---------|------------|----------|
| Q2 block t_total p50 | < 150ms | — | — | — | — | — |
| Hit rate plausibility | required < 80% | — | — | — | — | — |
| Inter-token reuse | > 0.1 | — | — | — | — | — |
| NLL delta (pilot) | — | < 1.0 (7B) | — | — | — | — |
| tok/s | — | — | >= cold ceiling | >= 10% over Ph2 | — | >= 0.1† |
| NLL/logit gate | — | — | NLL Δ < 1.0 | — | < 1.0 (4a,4c) / < 0.5 (4b) | — |
| Memory stable | — | — | RSS var < 500MB | RSS var < 500MB | — | No OOM† |
| App budget | — | — | — | — | — | <= 10.5 GB† |
| min_available_gb | — | — | — | — | — | > 1.0† |

*Phase 5 = provisional feasibility probe on 24 GB dev machine. †All Phase 5 metrics are informational indicators, not acceptance criteria. Final acceptance requires a run on actual 16 GB hardware.

**Note on Phase 2 tok/s target**: If Phase 0's `measured_inter_token_hit_rate < 0.1`, the warm-cache hypothesis is invalidated and Phase 2's target becomes the cold-cache ceiling from Phase 0 (likely ~0.10 tok/s). Phase 2 still runs to get actual end-to-end measurements.

## Benchmarks

All phases will log to `experiments.jsonl` with:
- `tok_s` — tokens per second (steady-state, excluding warmup)
- `ttft_ms` — time to first token (prefill latency)
- `peak_rss_mb` — peak resident set size
- `metal_active_mb` — active Metal/GPU memory (`mx.get_active_memory()`)
- `metal_peak_mb` — peak Metal/GPU memory (`mx.get_peak_memory()`)
- `available_gb` — system available memory (via psutil)
- `block_load_p50_ms` / `block_load_p95_ms` — per-block load latency
- `pagein_mb` / `pageout_mb` — page cache pressure (system-wide, directional only)
- `block_resident_fraction` — mean fraction of each streamed block's pages already resident before loading, measured via `mincore()` on a read-only sidecar mmap. Logged in Phases 0, 2, 3, and 5 for cross-phase comparison
- `ppl` / `ppl_delta` — perplexity metrics (Phase 4)

## Risks

1. **Q2 quality degradation**: 2-bit may be too aggressive. Mitigation: Phase 4 validates quality; fallback to Q3 for attention layers.
2. **Q4→Q2 re-quantization path**: Dequantizing Q4 then re-quantizing to Q2 introduces double quantization error. Better to start from the FP16 checkpoint if available. We'll measure this in Phase 4.
3. **QuantizedLinear bits mismatch**: Loading Q2 weights into a model initialized with Q4 modules requires updating the module's `bits` and `group_size`. Need to verify this works with MLX's QuantizedLinear.
4. **madvise may not help on Apple Silicon**: macOS may already optimize sequential reads from NVMe. Phase 3 will measure the actual impact.
5. **Cache coverage math assumes uniform block access**: In practice, some blocks may be accessed more frequently (e.g., early layers for prefix caching). This could help or hurt depending on the access pattern.

## Implementation Notes

### Q4 → Q2 Dequantization

MLX QuantizedLinear stores:
- `weight`: packed uint32 array (Q4: 8 values per uint32)
- `scales`: float16 per group
- `biases`: float16 per group

To dequantize: `mx.dequantize(weight, scales, biases, group_size=128, bits=4)` → float16 matrix.
Then re-quantize: `mx.quantize(float16_matrix, group_size=64, bits=2)` → new (weight, scales, biases).

### Safetensors Saving

Use the `safetensors` Python package to write Q2 tensors:
```python
from safetensors.numpy import save_file  # or safetensors.torch
tensors = {"model.layers.8.mlp.down_proj.weight": q2_weight, ...}
save_file(tensors, "q2_blocks_shard_00.safetensors")
```

### File Organization

```
~/.cache/huggingface/hub/models--local-moe--q2-blocks/
  q2_shard_00.safetensors   # blocks 8-20  (13 blocks, ~3.4 GB)
  q2_shard_01.safetensors   # blocks 21-33 (13 blocks, ~3.4 GB)
  q2_shard_02.safetensors   # blocks 34-46 (13 blocks, ~3.4 GB)
  q2_shard_03.safetensors   # blocks 47-59 (13 blocks, ~3.4 GB)
  q2_shard_04.safetensors   # blocks 60-71 (12 blocks, ~3.1 GB)
  model.safetensors.index.json  # index mapping block tensors to shards
```

Keep each shard under 4 GB to avoid mmap issues. 13 blocks × 262 MB ≈ 3.4 GB per shard (5 shards for 64 blocks).
