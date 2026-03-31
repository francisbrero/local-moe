# Experiment: Expert Offloading / SSD Streaming (Flash MOE style)

**Issue**: #2

## Hypothesis

On a 16GB M4 MacBook Pro, we can run MoE models significantly larger than what fits in memory by streaming expert weights from NVMe SSD on demand. With only K active experts per layer (~6-27MB per expert set depending on quantization), the OS page cache should provide acceptable hit rates, and serial GPU/SSD pipelining should avoid the unified memory contention that degrades concurrent access.

We expect to achieve >1 tok/s with a MoE model that would otherwise not fit in memory on 16GB.

## Approach

### Phase 0: Checkpoint Audit (scripts/checkpoint_audit.py)

Before running any benchmarks, inspect target MoE checkpoints to lock model parameters:
- Download checkpoint metadata (config.json, model index) for Qwen3-30B-A3B and Mixtral 8x7B
- Measure: total checkpoint size, non-expert weight size, expert weight size per expert, number of experts, top-k routing value
- Compute expert tensor shapes and dequant workspace requirements for 2/3/4-bit quantization
- Audit physical checkpoint layout: shard count, tensors per expert, byte contiguity (are experts stored contiguously or scattered across shards?)
- Determine the target model based on which fits within the 9GB model-side memory cap
- Lock parameters used by subsequent phases: expert tensor shapes, top-k, quantization format, storage layout
- Log audit results to `experiments.jsonl`

### Phase 1: NVMe Profiling (scripts/nvme_profile.py)

Profile raw M4 NVMe performance for expert-sized reads:
- Sequential `pread()` of 1MB, 2MB, 4MB, 8MB chunks (contiguous reads)
- Scattered reads simulating real checkpoint layout (multiple small tensors per expert across shard boundaries, using layout info from Phase 0)
- `mmap()` + `madvise(MADV_SEQUENTIAL)` vs explicit `read()`
- `fcntl(F_NOCACHE)` (bypass page cache) vs normal reads to measure cache contribution
- Measure bandwidth (GB/s) and latency (ms) for each approach
- Run under realistic memory pressure: allocate CPU ballast (force-touched with `memset` to ensure pages are faulted in) AND Metal buffer allocations (via MLX `mx.ones` or explicit `mx.eval` to force materialization) to simulate ~10-11GB usable. Verify actual pressure level with `psutil.virtual_memory().available` and `vm_stat` before accepting each run as representative.

### Phase 2: Page Cache Behavior (scripts/page_cache_bench.py)

Simulate expert access patterns and measure OS page cache effectiveness:
- Create a synthetic expert corpus sized to significantly exceed usable cache budget (target ~20-30GB, or use actual checkpoint files if available) so that eviction and refill behavior reflects the real streaming regime where the expert corpus is much larger than RAM
- Simulate MoE routing: access K=2-4 experts per "layer", with realistic expert popularity distributions (Zipf-like, as observed in MoE models)
- Measure cache residency using `mincore()` on mmap'd pages as the primary signal; use access latency as a secondary confirmation metric
- Track page-in/pageout deltas via `vm_stat` before and after each access round to detect OS memory pressure
- Test under varying memory pressure levels anchored to measured idle headroom: 10GB, 9GB, and 8GB available (12GB is not achievable on 16GB M4 after OS overhead). First measure actual idle `psutil.virtual_memory().available` to calibrate scenarios. Use force-touched CPU ballast and force-evaluated MLX allocations (`mx.eval`) to reach target pressure. Validate achieved pressure level with `psutil.virtual_memory().available` and `vm_stat` before each benchmark run.
- Compare along two independent dimensions:
  - **Access path**: `pread()`/`read()` vs `mmap()`
  - **Cache policy**: default vs `MADV_SEQUENTIAL` vs `MADV_RANDOM` vs `F_NOCACHE` (bypass)
  - Run each combination in both cold-cache (purge with `purge` command or `F_NOCACHE` pre-read) and warm-cache (repeat access) conditions

### Phase 3: Serial vs Concurrent GPU/SSD (scripts/gpu_ssd_contention.py)

Validate the 73% GPU throughput degradation claim on M4:
- Run a Metal compute shader (simple matmul via MLX) while simultaneously reading from SSD
- Compare: serial (read then compute) vs concurrent (read + compute overlapped)
- Measure GPU throughput (GFLOPS) and SSD throughput (GB/s) in each mode
- Run alongside realistic Metal/MLX memory allocations to model actual inference pressure
- This directly validates whether the serial pipeline is necessary on M4

### Phase 4a: Synthetic Expert Streaming Microbenchmark (scripts/expert_stream_synthetic.py)

Build a synthetic expert streaming microbenchmark. This phase validates: (1) the I/O streaming path overhead, (2) whether MLX can consume mmap-backed tensors without copying, and (3) determines the implementation path for Phase 4b.

- Create fake expert weight tensors on disk in quantized format, testing across 2-bit, 3-bit, and 4-bit (since 16GB may require 2-3 bit experts to fit within budget)
- **Zero-copy test**: Run a repeated loop (100+ iterations) of `mmap → mx.array wrap → dequant → matmul → discard`, tracking peak RSS, peak Metal memory (`mx.metal.get_peak_memory()`), and allocation growth across iterations. Only classify the MLX path as zero-copy if all three metrics stay flat (growth < 5% over the loop). A single RSS snapshot is insufficient — Metal-side staging or lazy copies may only manifest during compute. Document the result — this informs the Phase 4b implementation path but does not solely gate it.
- Include dequantization cost: benchmark the full pipeline of `read quantized expert → dequantize → matmul → discard`, not just plain GEMM, since dequant is a significant part of the real inference cost
- Benchmark: in-memory dequant+GEMM vs SSD-streamed dequant+GEMM (isolates streaming overhead)
- Implement the serial pipeline pattern: `read expert → dequant+GEMM → read expert → dequant+GEMM → ...`
- **Scope**: This is primarily an I/O-path and memory-behavior validation. The dequant+GEMM cost is included for realism.

### Phase 4b: Real MoE Integration (stretch goal / follow-up issue)

**Note**: Phase 4b is a substantial implementation effort (custom MoE loader, checkpoint surgery, potentially C/Metal staging buffers). The committed deliverable for Issue #2 is Phases 0-4a. Phase 4b will be attempted if time permits and promotion criteria are met; otherwise it becomes a follow-up issue.

**Promotion criteria**: Proceed only if Phase 4a shows the streaming overhead and memory budget are acceptable (streaming within 3x of in-memory, peak resident memory within budget). The implementation path depends on Phase 4a's zero-copy test result:
- If MLX zero-copy works: use MLX mmap-based tensor wrapping
- If MLX copies internally: use C/Metal staging buffers with explicit memory management
- Either path is viable — the zero-copy test determines HOW to implement, not WHETHER to proceed.
- **Constraint**: Pick ONE implementation path based on Phase 4a results. Do not attempt both in this issue.

If proceeding with full integration (using model and parameters locked in Phase 0 checkpoint audit):
- Write a custom loader for the chosen model: load non-expert weights via MLX, mmap-stream expert weights from safetensors on disk
- This requires lower-level MLX array manipulation or C/Metal code rather than `mlx_lm`

### Scope Clarification

Issue #2 focuses on **storage I/O, page cache behavior, and streaming architecture validation** — proving whether SSD-based expert streaming is viable on 16GB M4. Custom Metal dequant/matmul kernels are the domain of Issue #10 (Custom Metal Shaders). Phase 4a uses MLX's built-in dequant+matmul to model realistic compute cost, but writing hand-optimized Metal kernels is explicitly out of scope for this issue. The results from this experiment directly inform Issue #10's kernel design (e.g., whether to optimize for serial vs concurrent I/O, buffer sizes, etc.).

### Fallback Strategy

If Phase 4b proves too complex for this experiment:
1. The synthetic benchmark (Phase 4a) still validates the I/O streaming architecture
2. Phases 1-3 produce standalone benchmarks valuable for other approaches (issues #5, #9, #10)
3. Real MoE integration can become a follow-up issue building on these findings

## Canonical Evaluation Workload

All throughput and quality measurements use a fixed workload for reproducibility:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Target model | Determined by Phase 0 audit | Qwen3-30B-A3B preferred; Mixtral 8x7B fallback |
| Quantization | Determined by Phase 0 audit | 2-bit, 3-bit, or 4-bit based on memory fit |
| Context length | 2048 tokens | Matches existing harness |
| Prompt | `BENCH_PROMPT` from `scripts/prepare.py` | Distributed computing explanation |
| Max decode tokens | 256 | Matches existing harness |
| Sampling | Greedy (temperature=0) | Deterministic for reproducibility |
| Repetitions | 3 (warm runs) | Matches existing harness |
| Top-k routing | Model default | From Phase 0 checkpoint audit |
| Perplexity corpus | `PERPLEXITY_PASSAGES` from `scripts/prepare.py` | 4 passages: prose, code, qa, reasoning |
| Time budget | 300s per phase | Abort on timeout |

For Phase 4a synthetic benchmarks, the workload is defined by expert tensor shapes and batch sizes locked in Phase 0, with 100 simulated layer passes per measurement.

## Structured Logging

All phase scripts emit structured JSON records to `experiments.jsonl` using a schema compatible with the existing harness. Each record includes:
- `experiment_name`: e.g., `"nvme_profile_pread_4mb"`, `"page_cache_mmap_10gb"`, `"gpu_ssd_serial"`
- `phase`: `"checkpoint_audit"`, `"nvme_profile"`, `"page_cache"`, `"gpu_ssd_contention"`, `"expert_stream_synthetic"`, `"expert_stream"`
- `config`: phase-specific parameters (chunk size, pressure level, access path, quant bits, etc.)
- `results`: phase-specific measurements (bandwidth, latency, hit rate, throughput, etc.)
- `env`: hardware info, OS version, available memory at start
- `status`: `"completed"`, `"aborted_memory"`, `"aborted_timeout"`

## Metrics

All planned — none measured yet:

- [ ] SSD read bandwidth (GB/s) for various chunk sizes
- [ ] SSD read latency (ms) for expert-sized chunks
- [ ] Page cache residency rate (via `mincore`) under memory pressure
- [ ] Pageout/swap activity (via `vm_stat`) during benchmarks
- [ ] GPU throughput with/without concurrent SSD access (GFLOPS)
- [ ] Expert dequant+GEMM throughput: in-memory vs SSD-streamed (GFLOPS)
- [ ] MLX zero-copy verification (RSS delta on mmap tensor wrap)
- [ ] Peak dequant workspace (RSS delta during dequant+GEMM vs quantized expert size)
- [ ] Expert load latency p50/p95 (ms) — from SSD read to tensor ready
- [ ] Per-layer service time p50/p95 (ms) — read + dequant + matmul end-to-end
- [ ] tok/s (decode) with expert streaming (Phase 4b only)
- [ ] ttft (time to first token) with expert streaming (Phase 4b only)
- [ ] peak_rss_mb during inference
- [ ] peak Metal/GPU memory (via MLX memory tracking)
- [ ] `psutil.virtual_memory().available` floor during runs
- [ ] perplexity (Phase 4b only — must be within 0.5 of non-streamed baseline)

## Memory Budget

Resident allocations only — page cache is opportunistic and NOT counted as reserved:

| Component | Estimated Size | Notes |
|-----------|---------------|-------|
| Total RAM | 16 GB | M4 MacBook Pro |
| OS + apps | ~5 GB | Conservative estimate |
| **Available for inference** | **~11 GB** | Hard ceiling |
| Non-expert weights | TBD (measure) | Must inspect actual checkpoint |
| KV cache (2K context) | ~0.5-1 GB | Depends on quantization |
| Metal scratch buffers | ~0.5 GB | Command buffers, intermediates |
| Python/runtime overhead | ~0.5 GB | MLX, numpy, etc. |
| **Resident total (model-side cap)** | **TBD, target < 9 GB** | Leaves ~2 GB for page cache + VM headroom |
| Active experts (K=TBD, per layer) | TBD | Quantized size, streamed on demand; K from Phase 0 audit |
| Dequant workspace (per expert) | TBD | Dequantized expert + activation buffer; transient; measured in Phase 4a |
| Staging/double buffer (if used) | TBD | For read/compute overlap; transient |
| Page cache (opportunistic) | Whatever's left | OS manages; NOT guaranteed |

**Note on transient buffers**: During dequant+GEMM, the quantized expert expands ~4-8x (4-bit → fp16/fp32). These transient allocations must be tracked and included in peak memory measurements. Phase 4a must measure actual peak resident usage for the full dequant+GEMM path to validate the budget before Phase 4b.

**Phase 0 resolves the TBDs**: The checkpoint audit locks the target model and fills in measured values for non-expert weights, expert shapes, and dequant workspace. All subsequent phases use these measured values.

## Baselines

Three baselines for proper comparison:

1. **Dense in-memory baseline** (hardware reference): Largest dense model that fits (XL=14B or L=7B)
   ```bash
   # Edit scripts/benchmark.py: MODEL_TIER="XL", EXPERIMENT_NAME="baseline_XL"
   uv run python scripts/prepare.py --tier XL
   uv run python scripts/run.py
   ```

2. **Synthetic in-memory expert dequant+GEMM baseline**: Same expert tensor shapes and quantization as Phase 4a, but kept fully in memory. Isolates the cost of SSD streaming vs having experts resident.

3. **MoE in-memory baseline** (if Phase 4b proceeds): Same MoE model with all experts pinned in memory (on a machine with enough RAM, or a smaller MoE that fits on 16GB). This is the architecture-comparable baseline for tok/s and perplexity comparison. If no MoE fits fully in memory on 16GB, document this as the motivation for streaming.

## Success Criteria

### Prototype viability (minimum to continue)
1. **NVMe profiling**: >4 GB/s sequential read for 4MB+ chunks
2. **Page cache**: >40% residency (via `mincore`) under 10GB available with realistic expert patterns
3. **GPU/SSD contention**: Measured and documented (any result is useful)
4. **Synthetic streaming**: SSD-streamed expert dequant+GEMM within 3x of in-memory
5. **Zero-copy test**: MLX mmap tensor behavior documented (copy or no-copy); implementation path for Phase 4b determined

### Promotion to Phase 4b (worth building real integration)
1. **Page cache**: >60% residency under realistic pressure
2. **Synthetic streaming**: SSD-streamed expert dequant+GEMM within 2x of in-memory
3. **Memory budget**: model-side resident allocations stay within 9GB cap (including measured dequant workspace), with cache residency and memory headroom validated simultaneously during Phase 4a streaming runs
4. **No heavy swapping**: pageout delta < 100MB per benchmark run
5. **Checkpoint audit**: at least one MoE model's non-expert weights fit within measured budget

### Phase 4b success (if attempted)
1. **Throughput**: >1 tok/s with a model larger than available memory
2. **Memory safety**: `psutil.virtual_memory().available` stays above 1 GB throughout; peak Metal memory tracked and documented; pageout delta < 500MB per inference run
3. **Quality**: perplexity within 0.5 of MoE in-memory baseline (baseline 3). If no MoE fits fully in 16GB for a local baseline, use a fallback quality check: compare per-layer output logits against a reference run on a higher-memory machine, or compare against published perplexity numbers for the same model/quantization. Document which quality check was used and why.

## Rollback

Each phase produces standalone benchmark scripts. If expert streaming doesn't work:
- The NVMe/cache/contention benchmarks are still valuable for other approaches (issues #5, #9)
- Fall back to the existing MLX in-memory inference path
- Results inform memory budgets for other experiments
