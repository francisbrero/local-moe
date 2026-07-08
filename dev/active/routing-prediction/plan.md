# Experiment: Expert Routing Prediction + Prefetch (H2)

**Issue**: #22
**Hardware**: 24 GB M4 Pro MacBook Pro (~19-21 GB usable). Note: CLAUDE.md header says 16 GB but actual hardware is 24 GB M4 Pro (confirmed by all prior experiments and memory.md).

## Hypothesis

Expert routing decisions in MoE models are predictable enough (≥90% recall) that a lightweight predictor can enable async SSD prefetching to hide load latency. This experiment validates prediction viability and prefetch pipeline mechanics. The full throughput claim (60-80 tok/s) requires combining H2 prediction with C/Metal expert-addressable loading (#17) and faster kernels (#10) — that is a follow-up experiment.

## Approach

Three phases, building incrementally:

### Phase 1: Routing Pattern Analysis
- Hook into MLX model inference to capture router gate outputs per layer per token
- **Phase 1a (pilot)**: Run Qwen3-30B-A3B (4-bit) on a short prompt set (~200 tokens across 4 categories) to validate tracing hooks and get initial correlation signal
- **Phase 1b (full collection)**: Run on diverse prompt set (~10,000+ generated tokens across 20+ prompts spanning prose, code, QA, reasoning, multilingual, long-context) to build sufficient training corpus for Phase 2
- Log per-layer expert activation indices (top-K=8 out of 128 experts)
- Measure:
  - Cross-layer correlation: for each layer pair (N, N+L), what fraction of experts overlap?
  - Temporal locality: how often do consecutive tokens activate the same experts?
  - Expert co-occurrence: which experts tend to activate together?
  - Expert frequency distribution: Zipf-like or uniform?

**Implementation**: Python script `scripts/routing_trace.py` that:
1. Loads Qwen3-30B-A3B 4-bit via `mlx_lm` (full checkpoint loaded into unified memory)
2. Monkey-patches the MoE gate module to intercept:
   - Router logits before top-K selection (expert indices + gate weights)
   - Pre-attention hidden states (needed as predictor input in Phase 2)
3. Runs generation on 4 prompt categories (reuse from prepare.py prompt set)
4. Saves structured traces to `routing_traces.npz`:
   - Expert indices: (48 layers × num_tokens × 8) int16 — <1 MB for 200 tokens
   - Pre-attention hidden states: (48 layers × num_tokens × 2048) float16 — ~38 MB for 200 tokens
5. Computes and logs correlation/locality metrics to experiments.jsonl
6. Reports `peak_rss_mb` and `gpu_peak_memory_mb` (via `mx.metal.get_peak_memory()`)
7. **Overhead benchmark**: Runs the same prompt set with and without tracing hooks, reports traced_tok_s vs untraced_tok_s to quantify hook overhead

**Memory budget** (derived from H0 checkpoint audit: hidden_size=2048, 4 KV heads, head_dim=128, 48 layers):

*Phase 1a (pilot, ~200 tokens):*
- Model: full 4-bit checkpoint via `mlx_lm` ~14 GB in unified memory
- KV cache: 48 layers × 4 KV heads × 128 head_dim × 200 tokens × 2 bytes × 2 (K+V) = ~19 MB
- Trace buffer: expert indices <1 MB + hidden states ~38 MB = ~39 MB
- **Total persistent**: ~14.04 GB; **Peak transient**: ~14.3 GB (with eval scratch)

*Phase 1b (full collection, ~10,000+ tokens across 20+ prompts):*
- Model: same ~14 GB
- KV cache: ~950 MB worst case (10K tokens, but reset between prompts → ~50-100 MB per prompt)
- Trace buffer if accumulated: expert indices ~40 MB + hidden states ~1.9 GB = ~1.94 GB
- **Mitigation**: Flush traces incrementally to disk per-prompt (each prompt's traces written to `routing_traces_<prompt_id>.npz` before starting next prompt, KV cache reset). In-memory at any time: one prompt's traces only (~100 MB for ~500 tokens)
- **Total persistent**: ~14.1 GB; **Peak transient**: ~14.4 GB

Both within the 19-21 GB usable on the 24 GB M4 Pro.
- **Runtime memory gate**: log `available_gb` before/after model load, `peak_rss_mb`, `gpu_peak_memory_mb` via `mx.metal.get_peak_memory()`. Abort if available memory falls below 4 GB safety floor.

### Phase 2: Predictor Design + Training
- Build a cross-layer linear predictor (ETH Zurich / Fate approach):
  - Input: block-input hidden state for layer N (the residual stream tensor entering the transformer block, dimension 2048 for Qwen3-30B-A3B). This is available *before* layer N begins attention computation, making it compatible with "prefetch N+2 while computing N" — the predictor runs on the block-input, issues prefetch, then attention + MoE compute proceeds.
  - Output: predicted expert indices for layer N+L (L=1,2,3)
  - Architecture: 2 linear layers with ReLU, output → sigmoid → top-K
  - **Timing contract**: Predictor latency must be ≤ the attention compute time for layer N (~1-2ms estimated) so prefetch I/O can overlap with the remainder of layer N and all of layer N+1. Phase 2 measures this explicitly.
- Train on pre-attention hidden states (captured in Phase 1) paired with expert indices
- Evaluate prediction accuracy at lookahead L=1, 2, 3
- Also build a co-activation lookup table (BuddyMoE-style) for miss fallback

**Implementation**: Python script `scripts/routing_predictor.py` that:
1. Loads pre-attention hidden states and expert indices from Phase 1b traces (~10,000+ tokens across 20+ prompts)
2. Splits data by prompt into train (70%) / validation (15%) / test (15%) sets — no prompt appears in multiple splits, ensuring ~1500+ test tokens
3. Trains per-layer predictors via MLX (simple gradient descent, ~1 minute training)
   - Input: hidden states at layer N (float16, dim 2048)
   - Target: expert indices at layer N+L (multi-hot, 128 classes)
4. Evaluates on held-out test set: recall@K, precision@K, Jaccard overlap, exact set match per layer, per lookahead distance. Reports mean ± std across prompt groups.
5. **Predictor inference latency**: Measures per-token predictor execution time (all 48 layers) to quantify overhead added to the inference pipeline
6. Builds co-activation buddy table from trace statistics
7. Logs all metrics to experiments.jsonl (including `peak_rss_mb`, `gpu_peak_memory_mb`, predictor_latency_per_token_ms)

**Success gate**: mean recall@8 ≥ 90% at L=2 lookahead on held-out test set, with 95% confidence interval width ≤ 5pp (i.e., ≥90% of the 8 actually-activated experts are in the predicted set). Report mean ± std across prompt groups. Also report precision@8, Jaccard overlap, and exact set match for completeness. Based on prior art: 93-97% recall (ETH Zurich, Fate).

### Phase 3: Async Prefetch Pipeline (Synthetic I/O Simulation)
- **This is a synthetic I/O simulation**, not a real-weight prefetch test. The MLX-converted Qwen3 checkpoint does not expose experts as separately addressable tensors, and H0 Phase 4a showed MLX mmap wrapping is not zero-copy. Real expert-addressable prefetching requires the C/Metal integration from H0 Phase 4b (#17).
- Build a synthetic expert corpus preserving real checkpoint structure:
  - Use H0 checkpoint audit shard layout data to replicate real tensor placement (experts scattered across shards, 3 tensors per expert: gate_proj, up_proj, down_proj)
  - Preserve multi-shard expert crossings where they exist in the real layout
  - 48 layers × 128 experts × 2.25 MB = 13.5 GB of synthetic safetensors files
  - Stored on SSD, accessed via `os.pread()` at real offsets
  - **Note**: Even with realistic layout, this is still simpler than the real loading path (no safetensors header parsing overhead, no MLX copy semantics). Throughput results are labeled as upper bounds.
- Implement background SSD prefetch overlapping with simulated GPU compute:
  - Main thread: simulates GPU compute timing for layer N (using H0 Phase 4a measurements: H0 Phase 4a measured timing per expert)
  - Background thread: issues `pread()` calls for predicted expert tensors for layer N+2
  - On layer N+2 start: check if prefetched experts match actual routing → hit/miss
- Use `concurrent.futures.ThreadPoolExecutor` for async I/O
- Staging buffer: 8 experts × 2.25 MB = 18 MB (tiny overhead)
- Implement Least-Stale eviction for the staging cache (per SpecMD findings)

**Implementation**: Python script `scripts/routing_prefetch_bench.py` that:
1. Generates synthetic expert corpus (one-time setup, ~13.5 GB on SSD)
2. Replays real routing traces from Phase 1 through the prefetch pipeline
3. Background thread issues `pread()` calls for predicted experts from synthetic corpus
4. Main thread simulates GPU compute timing anchored to H0 Phase 4a measurements: compute_p50=0.26ms, load_p50=0.14ms, streamed_p50=0.40ms per expert (8 experts/layer = ~3.2ms/layer). Values read from `experiments.jsonl` at startup.
5. Measures: prefetch hit rate, I/O bandwidth, pipeline stall time, oracle throughput ceiling
6. Includes measured predictor latency (from Phase 2) and tracing overhead (from Phase 1) in timing model to report net projected benefit after H2 overhead
7. Compares: no-prefetch (baseline) vs perfect-prefetch (oracle) vs predicted-prefetch (with overhead)
7. Reports `peak_rss_mb`, `gpu_memory_mb`, pageins/pageouts via `vm_stat`

**Limitations**: This validates the prefetch pipeline mechanics and timing model but does NOT run actual model inference. Results are **upper bounds** — the synthetic corpus omits safetensors header parsing overhead and MLX copy semantics present in the real loading path. Real end-to-end throughput validation requires H0 Phase 4b (#17) for expert-addressable loading. The estimated tok/s from this phase is a projection, not a measurement.

## Metrics

- [x] recall_at_k (Phase 2) — fraction of actually-activated experts present in predicted set (primary gate metric)
- [x] precision_at_k (Phase 2) — fraction of predicted experts actually activated
- [x] jaccard_overlap (Phase 2) — intersection/union of predicted vs actual expert sets
- [x] exact_set_match (Phase 2) — fraction of tokens where predicted set exactly matches actual
- [x] cross_layer_correlation (Phase 1) — expert overlap between layers N and N+L
- [x] temporal_locality (Phase 1) — expert reuse across consecutive tokens
- [x] prefetch_hit_rate (Phase 3) — % of prefetched experts actually used
- [x] pipeline_stall_pct (Phase 3) — % time waiting for unprefetched experts
- [x] estimated_tok_s (Phase 3) — projected throughput upper bound (simulation only; not compared to real H0 baseline)
- [x] peak_rss_mb — all phases, measured via `resource.getrusage`
- [x] gpu_peak_memory_mb — all phases, measured via `mx.metal.get_peak_memory()` (reset before each phase)
- [x] gpu_active_memory_mb — secondary, via `mx.metal.get_active_memory()`
- [x] cache_hit_rate — staging cache effectiveness (Phase 3)
- [x] pagein_delta_mb / pageout_delta_mb — unified memory pressure (Phase 3)
- [ ] perplexity — N/A for this experiment (no model output modification; quality validated by confirming tracing hooks do not alter generation output in Phase 1)
- [ ] ttft — N/A (Phase 3 is simulation-only; real ttft requires #17)

## Throughput Ceiling Analysis

Using H0 Phase 4a measurements to bound what's possible:
- Per-expert compute: 0.26ms p50 (MLX GEMM)
- Per-expert SSD load: 0.14ms p50 (pread, warm cache)
- Per-expert streamed (load+compute): 0.40ms p50
- Per-token compute floor: 48 layers × 8 experts × 0.26ms = **99.8ms → ~10 tok/s** (compute-only, no I/O)
- Per-token streamed floor: 48 layers × 8 experts × 0.40ms = **154ms → ~6.5 tok/s** (serial load+compute)

**Key insight**: Even with perfect (oracle) prefetch that eliminates all SSD latency, the MLX Python path cannot exceed ~10 tok/s on this model. The issue's 60-80 tok/s target requires **both** routing prediction (H2, this experiment) **and** faster execution kernels (C/Metal, #17/#10). H2 alone validates that prediction is viable and quantifies the SSD latency contribution; the throughput uplift from H2 materializes only when combined with a faster compute path.

Phase 3 will compute and report this oracle ceiling explicitly.

## Baseline

From H0 experiments:
- Expert offloading without prediction: 6-20 tok/s (SSD-bottlenecked)
- SSD read latency: 0.31ms p50 (pread), 5.5-17 GB/s bandwidth
- GPU/SSD contention: 0.2% degradation (negligible on M4 Pro)
- Expert size at 4-bit: 2.25 MB each
- Page cache hit rate: 63-78% (Zipf access pattern)

```bash
# Phase 1
uv run python scripts/routing_trace.py

# Phase 2
uv run python scripts/routing_predictor.py

# Phase 3
uv run python scripts/routing_prefetch_bench.py
```

### Benchmark classes (all logged to experiments.jsonl)

| Class | Regime | Metric | Purpose |
|-------|--------|--------|---------|
| `h2_baseline_mlx` | Standard mlx_lm generation (no hooks) | tok/s, ttft, peak_rss_mb, gpu_peak_memory_mb | Baseline for overhead comparison |
| `h2_traced_mlx` | mlx_lm generation with tracing hooks | tok/s, ttft, peak_rss_mb, gpu_peak_memory_mb | Quantify Phase 1 hook overhead |
| `h2_predictor_latency` | Predictor inference on traced data | predictor_ms_per_token, peak_rss_mb | Quantify Phase 2 overhead |
| `h2_prefetch_sim` | Synthetic I/O pipeline replay | hit_rate, stall_pct, oracle_ceiling_tok_s | Phase 3 mechanism validation |

**Limitation**: The actual expert-offloading regime (SSD → GPU expert loading during inference) is not benchmarkable until #17 lands. The above classes measure H2 machinery overhead on the standard MLX path and simulate the offloading path. A matched before/after tok/s comparison in the real offloading regime is deferred to the follow-up experiment.

## Success Criteria

### This experiment (mechanism validation, simulation-only)
1. **Phase 1** (descriptive, no hard gate): Report cross-layer expert correlation, temporal locality, and frequency distribution. Also compute a cheap baseline: previous-layer expert recall (prior art: 78.8%). These inform Phase 2 but do not gate it — low raw overlap does not preclude a hidden-state predictor from achieving high recall.
2. **Phase 2**: mean recall@8 ≥ 90% at L=2 lookahead on held-out test set (95% CI width ≤ 5pp)
3. **Phase 3** (pipeline-mechanics validation only): Prefetch hit rate ≥ 85%, pipeline stall time ≤ 15% of total simulated time. Note: simulated tok/s is reported as informational upper bound only and is NOT used as evidence for the issue's throughput goal. Real throughput validation deferred to follow-up with #17.
4. **Quality sanity**: Phase 1 tracing hooks do not alter generation output (verified by comparing traced vs untraced outputs)

### Follow-up (real inference, depends on #17 + #10)
5. **End-to-end**: Measured tok/s ≥ 60 with prediction + C/Metal expert-addressable loading + faster kernels (validates the issue's 60-80 tok/s claim — requires both H2 prediction AND faster compute path)
6. **Perplexity**: No degradation vs baseline (confirms expert substitution on miss doesn't hurt quality)

If Phase 2 recall@8 at L=2 is < 70% on held-out test data, the prediction approach is not viable and we should document the negative result. Phase 1 correlation metrics are descriptive context, not a kill switch.

**Scope note**: This experiment validates prediction viability (can we predict experts?) and prefetch pipeline mechanics (can async I/O hide latency?). It does NOT validate the issue's headline 60-80 tok/s claim — that requires real expert-addressable loading (#17) and will be a separate follow-up experiment combining H2 prediction with H0 Phase 4b loading. This experiment's deliverables are: (a) trained predictors with measured recall, (b) co-activation buddy tables, (c) validated prefetch pipeline timing model — all reusable inputs for the follow-up.

## Rollback

Revert to H0 baseline (no prediction, on-demand expert loading). All H2 code is additive — no existing code is modified.
