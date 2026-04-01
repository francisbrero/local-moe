# Experiment: SSD-Offloaded Dense Inference with Layer LOD (H0+H5 Combo)

**Issue**: #28

## Hardware Scope Note

The project's `CLAUDE.md` targets a 16GB M4 MacBook Pro. This experiment targets a **24GB M4 Pro** because the 72B model is the specific goal of issue #28 (extending H5's finding that 72B at 2.0 bpw = 24.3 GB — just 0.3 GB over). The SSD streaming technique being validated here is equally applicable to the 16GB target — a successful Phase 1-2 on 7B validates the streaming pattern on any memory budget. Phase 3+ is 24GB-specific.

## Hypothesis

By combining SSD layer streaming (H0) with sensitivity-guided mixed precision (H5), we can run Qwen2.5-72B on a 24 GB M4 Pro by keeping only the critical first/last blocks resident in RAM at high precision while streaming the compressible middle blocks from NVMe on demand.

## Background

H5 (Layer LOD) showed 72B at 2.0 bpw = 24.3 GB — 0.3 GB over the 24 GB limit. No uniform bit-width fits. But H5 also proved that middle transformer blocks (60% of them) can be aggressively compressed with near-zero perplexity impact, while H0 proved NVMe streaming on M4 Pro is fast enough to serve weights on demand (0.4 ms per expert, 5.5-6.5 GB/s cold).

Key difference from H0: Dense models access ALL blocks sequentially for every token (not just 2/8 experts). This means cumulative streaming latency is the primary challenge.

## Approach

### Phase 0: Memory Budget Modeling

Compute exact per-block sizes for 72B at each bit-width and model the resident/streaming split:

1. **Per-block size computation**:
   - Download `config.json` and `model.safetensors.index.json` for Qwen2.5-72B
   - Compute exact tensor sizes per block at Q2/Q3/Q4
   - Account for quantization metadata (group scales + zero points)

2. **Resident set modeling**:
   - First 20% of blocks (16 blocks) at Q4
   - Last 20% of blocks (16 blocks) at Q4
   - Embeddings + lm_head at Q6
   - KV cache at FP16 (2048 context)
   - Non-model overhead: use measured value from H5 Phase 3 if available, else 2.0 GB estimate

3. **Streaming set modeling**:
   - Middle 60% of blocks (48 blocks) at Q2
   - These live on SSD, loaded via mmap on demand
   - Page cache acts as automatic LRU

4. **Feasibility gates** (full physical-memory budget, not just resident weights):
   - **24GB gate**: `pinned_total + OS_overhead ≤ 17 GB` — where pinned_total = resident weights + embeddings + lm_head + KV cache + peak Metal allocations + Python/MLX runtime. This leaves ≥7 GB for page cache + headroom.
   - **16GB gate (exploratory)**: Also compute whether a smaller model (e.g., 32B) or more aggressive streaming split could work on 16GB/~10-11GB usable. This informs future applicability to the project's canonical 16GB target.
   - If budget doesn't close, try adjusting: fewer resident blocks, Q3 for edges, compressed KV (H7), smaller context length
   - Document exact gap if it doesn't fit

**Metrics**: Per-block bytes at each bit-width, total resident bytes, total streaming bytes, headroom estimate (both 24GB and 16GB scenarios).

### Phase 1: Layer Streaming Prototype (on 7B)

Build a proof-of-concept that loads transformer blocks from disk on demand:

1. **Layer loading** (using loader path selected by Phase 1b):
   - **Expected path (C/pread)**: Use the staging buffer prototype from Phase 1b. Resident blocks loaded once into MLX at startup; streaming blocks loaded via pread into staging buffer per token.
   - **Fallback path (MLX mmap)**: Only if Phase 1b shows MLX zero-copy works (unlikely). Use mmap with `madvise` hints.
   - Track which blocks are "resident" (loaded at startup, kept in MLX memory) vs "streamable" (loaded per token from disk)

2. **Measurement on Qwen2.5-7B** (28 blocks), run in two memory regimes:

   **Regime A — Unpressured** (baseline, validates correctness):
   - **All-resident baseline**: Normal MLX inference, all blocks in RAM
   - **Stream-middle-only**: First/last 20% resident, middle 60% from mmap
   - **Stream-all**: Everything via mmap (worst case)
   - Measure: tok/s, peak RSS, per-block load latency (p50/p95/p99)

   **Regime B — Synthetic pressure** (simulates 72B memory environment):
   - **Parameterized from Phase 0 output**: Use Phase 0's computed `pinned_total + OS_overhead` for the chosen split to determine the target available memory: `target_available_gb = 24 - pinned_total - OS_overhead`
   - Allocate CPU+Metal ballast (using `experiment_utils.create_memory_pressure()`) to leave only `target_available_gb` free
   - Re-run stream-middle-only and stream-all under this pressure
   - This ensures page-cache warmup, eviction behavior, and tok/s are predictive of the actual 72B configuration
   - **Gate**: If pressured stream-middle-only degrades >3x vs unpressured all-resident, the streaming approach is unlikely to work at 72B scale

3. **Page cache residency measurement**:
   - Run 100 tokens of sequential generation
   - Use `mincore()` on each block's mmap region to measure per-block page residency (following `scripts/page_cache_bench.py` methodology from H0)
   - Use `vm_stat` deltas as a secondary system-wide pressure signal (pageouts/pageins)
   - Measure how quickly the page cache warms up (what % of blocks' pages are resident after N tokens?)

**Metrics**: tok/s, peak RSS (MB), per-block load latency (ms), per-block page residency via `mincore()` (%), system pageout/pagein delta (MB).

**Implementation**: `scripts/ssd_layer_stream.py` — reuses `layer_sensitivity.py` infrastructure + `experiment_utils.py` for logging.

### Phase 1b: Loader Strategy Selection (HARD GATE)

**Known blocker**: H0 Phase 4a already confirmed that MLX copies mmap data internally — `np.frombuffer(mmap)` + `mx.array()` causes full copies (229.9 MB RSS growth over 100 iterations, see `dev/active/expert-offloading/context.md`). This means **MLX cannot be used for zero-copy streaming** — every "streamed" layer would be copied into RAM, defeating the purpose.

This phase determines the implementation path before investing in throughput benchmarks:

1. **Reproduce H0's MLX copy finding** (quick confirmation, ~10 min):
   - Load 7B model, swap one layer 10 times, measure RSS growth
   - Expected: RSS grows ~2.3 MB per swap (confirming H0 result)
   - If somehow fixed in newer MLX: proceed with MLX path (unlikely)

2. **If MLX copies confirmed (expected path) — prototype C/pread loader**:
   - Implement a minimal C extension that loads safetensors blocks via `pread()` into a pre-allocated staging buffer
   - Staging buffer: single fixed-size Metal buffer (~200 MB for one 72B Q2 block)
   - Load flow: `pread()` → staging buffer → dequantize → feed to MLX model forward pass
   - This reuses the staging buffer across blocks (no RSS growth)

3. **mlock/residency validation**:
   - Probe macOS mlock limits: `getrlimit(RLIMIT_MEMLOCK)` and test actual lockable bytes
   - If mlock is capped (default 64 KB on macOS without root), use `madvise(MADV_WILLNEED)` as a hint-only fallback
   - Test under pressure: allocate ballast to ~14 GB, verify WILLNEED pages have higher residency than DONTNEED pages via `mincore()`
   - **Fallback if mlock unavailable**: Rely on OS page cache LRU behavior (resident blocks stay warm due to repeated access every token). This is weaker but may be sufficient.

4. **Per-layer file layout test**:
   - Test loading a model with weights split into per-layer safetensors files
   - Verify the C loader can independently open/pread individual layer files
   - Measure: pread latency for ~200 MB block from individual files

5. **End-to-end inference integration test** (critical — validates the full path):
   - Load 7B model normally, then replace one middle block's weights with data loaded via the C/pread staging buffer
   - Run a forward pass through the entire model including the swapped block
   - Verify: correct output (logits match baseline within numerical tolerance)
   - Verify: RSS and Metal memory stay flat (within ±5%) across 10 repeated load-swap-forward cycles
   - This gates the FULL path: `pread() → staging buffer → mx.array() → model forward → output`
   - If MLX copies the staging buffer data into a new tensor on each forward pass, RSS will still grow — this test catches that

**Gate (all conditions must pass)**:
- Flat RSS across 10 repeated load-swap-forward cycles (within ±5%)
- Flat Metal memory across 10 cycles (within ±5%)
- Correct output (logits match within tolerance)
- Per-block load latency within 2x of H0's measured pread bandwidth (cold: <75 ms for 200 MB)
- If any condition fails, evaluate whether the issue is fixable (e.g., explicit `mx.array` reuse, Metal buffer recycling) or fundamental. If fundamental, abort the experiment.

**Metrics**: RSS delta per load-swap-forward cycle (MB), Metal memory delta per cycle (MB), pread latency per block (ms), logit match tolerance, mlock achievable bytes, WILLNEED vs DONTNEED residency difference under pressure.

### Phase 2: Scheduling Strategy Selection

Dense model block ordering is deterministic (block 0 → 1 → 2 → ... → N-1), so we can potentially overlap loading and compute. However, H0 found concurrent GPU+SSD activity on unified memory can reduce throughput — so overlap is not assumed beneficial.

**The strategies below depend on the loader selected by Phase 1b:**

**If C/pread selected (expected path)**:
1. **Serial pread**: Load block N via `pread()` into staging buffer, compute, then load N+1
2. **Double-buffer pread**: Two staging buffers; background thread `pread()`s block N+1 into buffer B while GPU computes block N from buffer A
3. **Triple-buffer pread**: Three buffers; pread N+2 while computing N and N+1 is ready

**If MLX mmap selected (unlikely fallback)**:
1. **Serial mmap**: Touch block N pages, compute, then touch N+1
2. **One-block lookahead**: `madvise(MADV_WILLNEED)` for block N+1 during compute of N
3. **Two-block lookahead**: Prefetch N+1 and N+2 during compute of N

**Common measurements** (per strategy):
- Block load time (cold/warm), block compute time, GPU stall time
- Context length sweep: 128, 512, 2048 tokens to find compute/load crossover
- Unified memory contention: does overlap degrade GPU compute throughput?

**Gate**: Carry forward only the strategy that wins on end-to-end tok/s under realistic memory pressure (parameterized from Phase 0 — see below). If serial wins, that's fine — it simplifies the implementation.

**Metrics**: End-to-end tok/s per strategy, overlap efficiency (%), GPU stall time (ms), latency breakdown per block, memory pressure behavior.

### Phase 2b: 72B-Shaped Synthetic Streaming Benchmark

The 7B model's middle blocks are too small to stress the page cache the way 72B will. Before downloading 72B, run a synthetic benchmark that replicates the 72B streaming geometry:

1. **Create synthetic 72B-shaped corpus**:
   - Generate 48 files of ~200 MB each (matching 72B Q2 middle blocks) — total ~10 GB on disk
   - Fill with random data (content doesn't matter for streaming/cache benchmarks)
   - Access them sequentially (block 0 → 1 → ... → 47) in a loop simulating token generation

2. **Run under 72B memory pressure** (parameterized from Phase 0):
   - Use Phase 0's chosen split to determine: number of streaming blocks, block size, and `target_available_gb`
   - Allocate ballast to leave only `target_available_gb` free
   - Access files sequentially using the Phase 1b loader (pread or mmap) for 50 "token" iterations
   - Measure per-block access latency (p50/p95/p99), `mincore()` residency per block, pageout rate

3. **Cache-thrash detection**:
   - The 10 GB streaming set competing for ~4-6 GB of hot cache should cause eviction
   - Measure: does residency stabilize (steady-state LRU) or oscillate (thrashing)?
   - Record steady-state cache hit rate after warmup

**Gate**: If steady-state per-block access latency p95 > 50 ms (worse than cold SSD read), the system is thrashing and 72B streaming is not viable without reducing the streaming set or increasing available cache. Document the gap and possible mitigations.

**Metrics**: Per-block latency (p50/p95/p99), page residency per block, pageout rate (MB/s), thrash detection (residency stability), steady-state cache hit rate.

### Phase 3: 72B Integration Test

The main event — actually running 72B on 24 GB:

1. **Model preparation**:
   - Download Qwen2.5-72B-Instruct at mixed precision
   - Use OptiQ or manual allocation: Q4 for first/last 20%, Q2 for middle 60%
   - Store the model with resident/streaming blocks clearly separated

2. **Inference test** (using loader and scheduling strategy selected by Phase 1b/2):
   - Load resident blocks into RAM at startup
   - Stream middle blocks from SSD using the Phase 1b loader (expected: C/pread staging buffers)
   - Use the Phase 2 winning scheduling strategy
   - Generate 100 tokens and measure tok/s, peak RSS, perplexity

3. **Quality validation** (two reference targets):
   - Perplexity on WikiText-2 subset (32 sequences × 2048 tokens)
   - **Primary quality control** (streaming artifact isolation): Before streaming, load the 72B mixed-precision checkpoint all-resident on this same machine by temporarily disabling background apps and using `MADV_SEQUENTIAL` hints to let the OS page naturally. Run the same WikiText-2 eval. If the model cannot load all-resident even briefly (OOM), run the eval on a subset of blocks (first 40 blocks only, measuring partial-model perplexity) and compare the same partial-model eval under streaming. This isolates streaming artifacts from quantization effects using a reproducible local reference.
   - **Product comparison**: Compare against Qwen2.5-14B at Q4 (current best that fits without streaming). This answers: does 72B-streaming beat 14B-resident on quality?

4. **Memory stability test**:
   - Generate 500 tokens continuously
   - Monitor `psutil.virtual_memory().available` — must stay above 1 GB
   - Monitor pageout rate — excessive pageouts = OOM risk

**Metrics**: tok/s, peak RSS, peak Metal memory, perplexity (WikiText-2), available memory floor, pageout rate.

### Phase 4: Optimization

If Phase 3 works at all, optimize for usability:

1. **Tune the resident/streaming split**:
   - Use H5 sensitivity data to find the optimal cut point
   - Maybe first 10% + last 10% is enough at Q4 (saving more RAM for page cache)
   - Or keep last 3 blocks at Q8 (most sensitive per H5 data)

2. **Stack with KV cache compression** (H7):
   - If KV cache uses 2-4 GB at 2048 context, compressing it frees RAM for more resident blocks
   - Test with TurboQuant-style Q4 KV cache

3. **Reduce GPU stall time**:
   - Measure GPU stall time: `total_wall_time - sum(per_block_compute_time)` per token
   - Use `mx.metal.get_peak_memory()` to track GPU memory pressure
   - If stall time is high, try: deeper prefetch, smaller resident set (more cache budget), or async block loading via a background thread

**Metrics**: tok/s improvement per optimization, memory headroom gained, GPU stall time reduction (%).

## Hardware

- Apple M4 Pro, 24GB unified memory
- ~19-21GB usable after OS overhead
- NVMe SSD: 5.5-6.5 GB/s cold, 15-17 GB/s warm (from H0)

## Memory Budget (72B Target)

**IMPORTANT**: The table below uses rough estimates. Phase 0 MUST replace these with exact per-block byte counts computed from `model.safetensors.index.json`. The 16/48/16 Q4/Q2/Q4 split yields ~2.8 bpw average (NOT the 2.0 bpw from H5's uniform Q2 estimate of 24.3 GB). This means total model bytes are higher than 24.3 GB — the streaming approach is what makes it viable (only the resident portion must fit in RAM).

**Back-of-envelope sanity check** (Phase 0 will compute exact values):
- 72B model at 4 bpw (Q4) ≈ 36 GB total trunk. Per block (80 blocks): ~450 MB/block
- 72B model at 2 bpw (Q2) ≈ 18 GB total trunk. Per block: ~225 MB/block
- 32 resident blocks at Q4: 32 × 450 MB ≈ **14.4 GB** (not 8 GB!)
- 48 streaming blocks at Q2: 48 × 225 MB ≈ **10.8 GB** on SSD

This means resident blocks alone may be ~14 GB — making the budget very tight. Phase 0 must explore mitigations if this doesn't close:
- Fewer resident blocks (e.g., 10%+10% = 16 blocks instead of 32)
- Lower precision for edge blocks (Q3 instead of Q4)
- KV cache compression (H7) as a prerequisite, not an optimization
- Smaller context length

Full physical-memory accounting — on Apple Silicon, all components share the same physical RAM pool:

| Component | Rough Estimate | Category | Notes |
|-----------|---------------|----------|-------|
| Resident blocks (32 at Q4) | ~14 GB (**Phase 0 will refine**) | Pinned | First/last 20% of 80 blocks — MAY NEED REDUCTION |
| Embeddings + LM head (Q6) | ~2 GB | Pinned | `vocab_size=152064 × hidden_size=8192` |
| KV cache (2K context, FP16) | ~2 GB | Pinned | 80 layers × GQA heads |
| Peak Metal allocations | ~0.5 GB | Pinned | Command buffers, intermediates, scratch |
| Python/MLX runtime | ~1.5 GB | Pinned | Interpreter, MLX engine, Metal driver |
| **Total pinned (must fit)** | **~20 GB** (**likely must shrink**) | — | Non-evictable |
| OS + background apps | ~3 GB | Pinned | Conservative estimate |
| **Total pinned + OS** | **~23 GB** | — | **DOES NOT CLOSE at 20/60/20 Q4/Q2/Q4** |

**Phase 0 must find a split that closes**. Candidate mitigations (to be evaluated in Phase 0):

| Mitigation | Resident savings | Trade-off |
|-----------|-----------------|-----------|
| 10%+10% resident (16 blocks at Q4) | ~7 GB saved | More blocks streamed, higher latency |
| Q3 for edge blocks instead of Q4 | ~3.5 GB saved | Slight quality impact on edges |
| KV cache at Q4 (H7) | ~1 GB saved | Needs TurboQuant integration |
| 512 context instead of 2048 | ~1.5 GB saved | Much shorter context |
| Combination of above | ~10+ GB saved | Multiple trade-offs |

**If Phase 0 cannot find any split where `pinned_total + OS ≤ 17 GB`**, the experiment fails fast with documented analysis of the gap and what hardware budget would be needed.

**Cold-cache fallback target**: If page cache is insufficient and hit rates are low, the cold-cache tok/s (all blocks from SSD) becomes the acceptance floor. Phase 2 benchmarks serial cold-cache tok/s to establish this floor.

## Success Criteria

### Phase 0 (gate)
- Exact per-block sizes computed from actual checkpoint metadata
- Full physical-memory budget: `pinned_total + OS_overhead ≤ 17 GB` (leaves ≥7 GB for page cache + headroom)

### Phase 1 (minimum viable)
- Layer streaming prototype works on 7B (unpressured regime validates correctness)
- Unpressured: stream-middle-only adds <50% latency vs all-resident
- Pressured (simulated 72B cache budget): stream-middle-only degrades <3x vs unpressured all-resident
- Page residency via `mincore()` >50% after 10 tokens (unpressured)

### Phase 2 (scheduling strategy selection)
- Identify the winning scheduling strategy (serial, 1-block lookahead, or 2-block lookahead) under realistic memory pressure
- If a prefetch variant wins: report overlap efficiency metrics
- If serial wins: record that as the accepted outcome and proceed with serial
- Winning strategy achieves ≥0.3 tok/s on 7B under pressure (floor for 72B viability)

### Phase 3 (the big test)
- 72B actually loads and generates on 24 GB M4 Pro
- Peak RSS stays under 22 GB, `psutil.virtual_memory().available` stays above 1 GB
- Peak Metal memory (`mx.metal.get_peak_memory()`) within budget from Phase 0
- Generates at ≥1 tok/s (cold-cache floor; warm-cache target higher)
- **Quality control**: Perplexity within 5% of local non-streamed reference (all-resident or partial-model, same evaluation harness)
- **Product comparison**: Perplexity better than Qwen2.5-14B at Q4 on WikiText-2

### Phase 4 (optimization)
- tok/s improves to ≥2 tok/s
- Can generate 2048+ tokens without OOM
- GPU stall time (time GPU is idle waiting for block loads) reduced by ≥30% vs Phase 3 baseline. Measured as: `total_wall_time - sum(per_block_compute_time)` averaged over 100 tokens.

## Risks

1. **Cumulative streaming latency**: 80 blocks × 36 ms = 2.9s per token without prefetch. Even with prefetch, might be 0.5-1s per token — barely interactive.
2. **Page cache eviction under pressure**: 24 GB total, 14 GB resident, 10 GB streaming = OS under constant pressure. May thrash.
3. **MLX mmap control**: MLX loads safetensors via mmap, but we may not be able to control which pages are pinnable vs evictable.
4. **72B sensitivity differs from 7B**: The U-shape was confirmed on 7B but magnitudes may differ at 72B scale.
5. **Dense ≠ MoE for streaming**: H0 tested expert offloading where only 2/8 experts load per layer. Dense needs ALL blocks, every token. Fundamentally different access pattern.

## Baselines

Reproducible commands for baseline measurements used as references throughout the experiment:

```bash
# 7B all-resident baseline (Phase 1 reference)
uv run python scripts/ssd_layer_stream.py --model mlx-community/Qwen2.5-7B-Instruct-4bit --mode all-resident --n-tokens 100

# 7B streaming under pressure (Phase 1 Regime B — pressure-gb from Phase 0 output)
uv run python scripts/ssd_layer_stream.py --model mlx-community/Qwen2.5-7B-Instruct-4bit --mode stream-middle --pressure-gb $TARGET_AVAILABLE_GB --n-tokens 100

# 14B Q4 baseline (Phase 3 product comparison)
uv run python scripts/layer_lod_bench.py --model mlx-community/Qwen2.5-14B-Instruct-4bit --eval-only

# 72B-shaped synthetic streaming (Phase 2b — params from Phase 0 output)
uv run python scripts/ssd_synthetic_stream.py --n-blocks $N_STREAMING_BLOCKS --block-size-mb $BLOCK_SIZE_MB --pressure-gb $TARGET_AVAILABLE_GB --iterations 50
```

Note: `$TARGET_AVAILABLE_GB`, `$N_STREAMING_BLOCKS`, `$BLOCK_SIZE_MB` are outputs of Phase 0's memory budget computation.

## Structured Logging (`experiments.jsonl`)

All scripts emit structured JSON to `experiments.jsonl` via `scripts/experiment_utils.py`:

| Field | Description | Phases |
|-------|-------------|--------|
| `experiment_name` | e.g., `ssd_lod_memory_budget_72b`, `ssd_lod_stream_7b` | All |
| `phase` | `memory_budget`, `layer_stream`, `mmap_validation`, `prefetch`, `integration_72b`, `optimization` | All |
| `config.model` | Model ID | All |
| `config.quant_split` | e.g., `{resident: "Q4", streaming: "Q2", split: "20/60/20"}` | 1-4 |
| `config.context_length` | Tokens | 1-4 |
| `results.tok_s` | Decode throughput | 1-4 |
| `results.peak_rss_mb` | Peak RSS | 1-4 |
| `results.peak_metal_mb` | Peak GPU memory via `mx.metal.get_peak_memory()` | 1-4 |
| `results.perplexity` | WikiText-2 perplexity | 3-4 |
| `results.block_residency_pct` | Per-block page residency via `mincore()` | 1-2 |
| `results.cache_hit_rate` | Fraction of block accesses served from page cache (pages resident before first touch per `mincore()`) | 1-2b |
| `results.pageout_delta_mb` | System pageout delta during run (via `vm_stat`) | 1-3 |
| `results.block_latency_p50_ms` | Per-block load latency | 1-2 |
| `results.block_latency_p95_ms` | Per-block load latency | 1-2 |
| `results.available_memory_floor_gb` | Min available memory during run | 3-4 |
| `results.pageout_rate_mb_s` | Pageout rate during generation | 3-4 |
| `results.prefetch_overlap_pct` | % of SSD latency hidden by compute | 2 |
| `results.pass_fail` | Pass/fail vs phase success criteria | All |

## Dependencies

- Existing scripts: `scripts/layer_sensitivity.py`, `scripts/experiment_utils.py`
- Existing infra: `scripts/nvme_profile.py`, `scripts/page_cache_bench.py` (from H0)
- Models: Qwen2.5-7B (Phase 1-2), Qwen2.5-72B (Phase 3-4)
- Python: `mlx`, `mlx-lm`, `psutil`, `numpy`

## Rollback

Each phase is independently valuable:
- Phase 0 memory modeling is useful for any future 72B attempt
- Phase 1 layer streaming prototype is reusable for other large model experiments
- Phase 2 prefetch patterns apply to any sequential weight loading
- Phase 3 negative results (too slow) would still inform project direction
