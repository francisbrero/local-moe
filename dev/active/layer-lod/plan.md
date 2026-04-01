# Experiment: Layer LOD — Variable Precision by Depth

**Issue**: #25

## Hypothesis

Mixed-precision quantization (Q4 for high-importance first/last layers, Q2 for low-importance middle layers) achieves better quality than uniform quantization at the same memory budget. The primary target is fitting larger models or getting better quality at the 14B scale. Fitting a 70B model in 24GB without SSD streaming is a stretch goal that depends on whether the strict memory fit formula closes — the experiment will quantify the actual gap if it doesn't.

## Approach

### Phase 0: Capability Spike + Analytical Feasibility Gate

This phase is a hard gate — do not proceed to Phase 1 unless all three checks pass.

#### 0a. Toolchain validation

The "mlx-optiq" / "OptiQ" ecosystem may exist as pre-quantized HuggingFace checkpoints (e.g., `mlx-community/Qwen3.5-2B-OptiQ-4bit`) rather than a pip-installable tool. MLX's built-in `mlx_lm.convert` supports fixed mixed-precision recipes (`mixed_2_6`, `mixed_3_4`, `mixed_3_6`, `mixed_4_6`) but may not support arbitrary per-layer allocations.

1. **Check OptiQ availability**: Search PyPI, GitHub, and HuggingFace for an installable `mlx-optiq` or `optiq` package. If it exists, install and test. If not, document what's actually available.
2. **Check MLX mixed-quant capability**: Test what `mlx_lm.convert` supports:
   - Can we specify per-layer bit-widths via config? (Check `mlx_lm` source for `bits_per_weight` or similar per-layer config)
   - Can we load pre-quantized OptiQ checkpoints from HuggingFace? (Try `mlx-community/*-OptiQ-*`)
   - Can we manually edit quantization config after conversion to change per-layer assignments?
3. **Determine the implementation path**:
   - **Path A**: OptiQ tool exists → use it directly for per-layer allocation
   - **Path B**: MLX supports per-layer config → write our own allocation script using Phase 1 sensitivity data + MLX convert
   - **Path C**: Neither works → build a custom converter that re-quantizes individual layers at different bit-widths and reassembles the checkpoint
   - **Path D**: None feasible → abort experiment with documented findings
4. **Proof of concept**: Load a small model (Qwen2.5-3B) with at least two different bit-widths across layers. Confirm generation works. This is the minimum bar to proceed.

#### 0b. Tensor-level memory accounting

Compute exact memory requirements from checkpoint manifests (no guessing):

1. Download `config.json` for Qwen2.5-7B, Qwen2.5-14B, and Qwen2.5-72B from HuggingFace
2. For each model, compute tensor-by-tensor budget:
   - **Trunk weights** (transformer layers): `num_hidden_layers × params_per_layer × avg_bits / 8`
   - **Quantization metadata**: group scales + zero points (typically 0.5-1 bit/param extra)
   - **Embeddings** (`embed_tokens`): `vocab_size × hidden_size × embed_bits / 8` — note Qwen2.5 uses **untied embeddings** (`tie_word_embeddings: false`)
   - **LM head** (`lm_head`): `vocab_size × hidden_size × head_bits / 8` — separate from embeddings
   - **KV cache**: `2 × num_layers × num_kv_heads × head_dim × context_len × kv_bits / 8`
   - **Non-model overhead** (single term, avoids double-counting on unified memory): Use 2.0 GB initial estimate. Measured as: `peak_process_memory - (trunk_bytes + quant_meta_bytes + embed_bytes + lm_head_bytes + kv_bytes)`. This captures Metal allocator, command buffers, Python runtime, and MLX intermediates as one non-overlapping value. Refined after Phase 0c baseline load, and again after Phase 3.
3. Compute fit for each model at each average trunk bit-width (2.0, 2.5, 3.0, 3.5, 4.0)
4. **Provisional 72B feasibility verdict**: If Qwen2.5-72B does NOT fit at any bit-width even with the optimistic 2.0 GB `non_model_overhead` estimate, document the gap and downgrade Phase 4 to "analysis-only with SSD streaming recommendation." If it fits with estimated overhead but is tight (<2 GB headroom), mark as "provisional — revisit after Phase 3 measured `non_model_overhead`." The **final 72B go/no-go** is deferred to after Phase 3, when the measured `non_model_overhead` replaces the estimate.
5. Log all accounting to `experiments.jsonl`

#### 0c. Model selection and baselines

1. **Pick test models** based on 0b accounting:
   - **Small model (development)**: Qwen2.5-7B — fast iteration
   - **Target model**: Qwen2.5-14B (fits in 24GB at Q4)
   - **Stretch**: Qwen2.5-72B (only if 0b shows it can fit)
2. Download uniform Q4 and Q2 versions of the small model
3. Measure baseline perplexity, peak RSS, peak Metal memory, available memory floor, and **non-model overhead** = `peak_process_memory - (deterministic_model_bytes + kv_bytes)`. This single measured value replaces the Phase 0b overhead estimate and is carried forward to the 72B fit formula.

### Phase 1: Layer Sensitivity Profiling (scripts/layer_sensitivity.py)

Profile per-layer quantization sensitivity on the small model:

1. Load the small model at FP16 or Q8 (highest precision that fits)
2. For each layer i (0 to N-1):
   - Quantize only layer i to Q2 while keeping all others at Q4
   - Measure perplexity on a calibration corpus (WikiText-2 or similar)
   - Record: layer index, perplexity delta vs baseline, layer type (attention vs FFN)
3. Plot the sensitivity curve — expect U-shape (high at edges, low in middle)
4. Also measure with Q3 substitution (not just Q2) to understand the sensitivity gradient
5. Log all results to `experiments.jsonl`

**Alternative (faster)**: If per-layer requantization is too slow, use KL-divergence between full-precision and quantized layer outputs as a proxy for sensitivity (no full-model perplexity eval needed per layer). If Phase 0 found an OptiQ tool with built-in sensitivity analysis, use that instead.

**Metrics**:
- Per-layer perplexity delta (Q2 substitution)
- Per-layer perplexity delta (Q3 substitution)
- Sensitivity curve shape (confirm/deny U-shape)
- Layer type breakdown (attention vs FFN sensitivity)
- Peak Metal memory (`mx.metal.get_peak_memory()`) per run
- Peak RSS per run

### Phase 2: Optimal Allocation Search (scripts/optiq_allocate.py)

Using the toolchain identified in Phase 0, compute optimal per-layer bit allocations.

**Budget-matched comparison methodology**: All strategies must be compared at matched memory budgets. Define target budgets as exact total checkpoint bytes (trunk weights + quant metadata, excluding fixed-cost components like embeddings/head/KV which are identical across strategies). For each target budget, derive allocations that match within ±2% tolerance:

- **Target budgets**: 3 fixed points corresponding to effective average trunk bpw of 2.5, 3.0, 3.5 (plus uniform Q2 and Q4 as anchors)
- Every strategy at each budget must produce the same total trunk weight bytes (±2%)

1. **If OptiQ/automated tool available (Path A from Phase 0)**:
   - Run it on the small model constrained to each target budget
   - Record per-layer allocations and compare with Phase 1 sensitivity curve
2. **If manual allocation (Path B/C from Phase 0)**:
   - Use Phase 1 sensitivity data to assign bit-widths: highest-sensitivity layers get Q4, lowest get Q2
   - Implement a greedy knapsack solver: sort layers by sensitivity, assign highest bits to most sensitive layers until budget is exhausted
   - Generate allocations for each target budget
3. For each strategy at each matched budget, quantize and measure perplexity
4. Strategies to compare (all at each matched budget):
   - **Sensitivity-guided**: Greedy knapsack from Phase 1 data
   - **U-shape**: Q4 for first/last 20% of layers, Q2 for middle 60% (adjust proportions to match budget)
   - **Gradient**: Q4 → Q3 → Q2 → Q3 → Q4 (smooth ramp, adjusted to match budget)
   - **Uniform**: Single bit-width matching the budget (e.g., uniform Q3 for 3.0 bpw budget)
5. Log all results to `experiments.jsonl`

**Metrics** (per strategy, per matched budget):
- **Primary memory metric (deterministic)**: `trunk_weight_bytes`, `quant_metadata_bytes`, `embed_bytes`, `lm_head_bytes`, `kv_bytes`, `total_checkpoint_bytes` — computed from actual file/tensor sizes
- Perplexity on WikiText-2
- **Supporting system metrics**: Peak RSS (MB), peak Metal memory (`mx.metal.get_peak_memory()`), `psutil.virtual_memory().available` floor
- Sensitivity-guided vs hand-crafted U-shape comparison (correlation coefficient)

### Phase 3: Benchmark Suite (scripts/layer_lod_bench.py)

Full benchmark comparing allocations on the target model (14B). All strategies are compared at matched memory budgets (same methodology as Phase 2 — ±2% trunk weight bytes tolerance).

1. For each allocation strategy at each matched budget:
   - Measure perplexity on WikiText-2 (hard gate metric)
   - Measure tok/s (decode throughput)
   - Measure TTFT
   - Run directional task quality evaluation (prose, code, math, QA)
2. Generate a Pareto frontier plot: perplexity vs **deterministic checkpoint bytes** for each allocation
3. Identify the best allocation for our 24GB budget
4. **Measure non-model overhead** for Phase 4 gate: compute `non_model_overhead = peak_process_memory - (deterministic_model_bytes + kv_bytes)` from these runs. This measured value replaces the Phase 0 estimate in the 72B fit formula.

**Metrics** (per strategy, per matched budget):
- **Primary memory metric (deterministic)**: `trunk_weight_bytes`, `quant_metadata_bytes`, `embed_bytes`, `lm_head_bytes`, `kv_bytes`, `total_checkpoint_bytes`
- Perplexity on WikiText-2
- tok/s (decode)
- TTFT (ms)
- Directional task scores (pass@1, exact match — with confidence intervals)
- **Supporting system metrics**: Peak RSS (MB), peak Metal memory (MB), `psutil.virtual_memory().available` floor
- Memory savings vs uniform Q4 at matched perplexity (%)

### Phase 4: Large Model Deployment (stretch — analysis-first)

If Phase 3 shows clear benefit, evaluate whether a 70B model can fit:

1. **Tensor-level fit analysis** (reuses Phase 0b methodology, no shortcut formulas):
   - Use the same tensor-by-tensor accounting from Phase 0b for Qwen2.5-72B, ideally from actual safetensors index metadata (`model.safetensors.index.json`) rather than config-derived estimates
   - **Trunk weights**: sum per-layer tensor sizes at the proposed mixed-precision allocation (not `num_params * avg_bits / 8`)
   - **Quant metadata**: sum group scales + zero points per tensor from the actual quantization format
   - **Embeddings + LM head**: exact byte sizes from checkpoint manifest (Q6)
   - **KV cache**: `2 × num_layers × num_kv_heads × head_dim × context_len × kv_bits / 8` (using actual 72B config values)
   - **Non-model overhead**: **measured value from Phase 3** 14B runs (single term: `peak_process_memory - model_bytes - kv_bytes`, not estimated)
   - **Strict fit formula** (same canonical formula used throughout): `trunk_bytes + quant_meta_bytes + embed_bytes + lm_head_bytes + kv_bytes + non_model_overhead <= measured_available_memory`
2. If the formula closes with margin (>1GB headroom):
   - Download the model and attempt loading
   - Measure actual perplexity, tok/s, peak RSS, peak Metal memory
   - Compare against uniform Q2 70B published numbers
3. If the formula does NOT close:
   - Document the exact gap (e.g., "70B at 2.5 avg bits = 23.1 GB, budget = 21 GB, gap = 2.1 GB")
   - Compute what average bit-width WOULD be needed
   - Evaluate if combining with SSD streaming (H0) could bridge the gap
   - This analysis is still a valuable outcome

**Model family constraint**: Phase 4 must use **Qwen2.5-72B only** (same family as Phases 1-3). Do not switch to Llama-3.1-70B, as sensitivity profiles are not portable across architectures.

**Promotion criteria** (must pass before attempting Phase 4):
- Phase 3 shows >5% perplexity improvement of LOD over uniform at same matched memory budget (WikiText-2)
- U-shape confirmed on 7B (Phase 1), plus a lightweight 14B validation: profile sensitivity on a sampled subset of layers (first 3, middle 3, last 3) to verify the pattern transfers. If 14B validation is infeasible (too slow), accept 7B evidence plus Phase 3 benchmark outcomes as sufficient.
- Canonical strict fit formula closes with >1GB headroom using **measured** `non_model_overhead` from Phase 3 (not Phase 0 estimates)
- No showstopper tooling issues

## Hardware

- Apple M4 Pro, 24GB unified memory
- ~19-21GB usable after OS overhead
- NVMe SSD (not needed if model fits in RAM — that's the whole point)

## Memory Budget

All values below are estimates — Phase 3 runs will measure actual values to populate the strict fit formula for Phase 4.

| Component | Estimated Size | Notes |
|-----------|---------------|-------|
| Total RAM | 24 GB | M4 Pro MacBook Pro |
| OS + apps | ~3 GB | Conservative estimate |
| **Available for inference** | **~21 GB** | Hard ceiling; measured via `psutil.virtual_memory().available` |
| Trunk weights (14B, 40 layers) | 4-9 GB | Depends on average bit-width; excludes embed/head |
| Quantization metadata | ~0.5-1 GB | Group scales, zero points (~0.5-1 bit/param) |
| Embeddings at Q6 (untied) | ~0.7 GB | Qwen2.5: `vocab_size=152064 × hidden_size=5120` = 778M params, untied (`tie_word_embeddings: false`) |
| LM head at Q6 (separate) | ~0.7 GB | Same shape as embeddings, separate weight tensor |
| KV cache (2K context, kv_bits=4) | ~0.5-1 GB | Standard for 14B model |
| Non-model overhead (single term) | ~2.0 GB (initial est.) | Measured as `peak_process_memory - model_bytes - kv_bytes`. Captures Metal allocator, command buffers, Python/MLX runtime. Avoids double-counting on unified memory. Refined after Phase 0c and Phase 3. |

**Canonical strict fit formula** (used consistently everywhere, including Phase 4 gate):
`trunk_bytes + quant_meta_bytes + embed_bytes + lm_head_bytes + kv_bytes + non_model_overhead <= measured_available_memory`

Where `non_model_overhead` is a single measured term (see below).

## Canonical Evaluation Workload

### Compression Quality (Perplexity)

Fixed corpus for all phases — measures pure compression quality:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Perplexity corpus | WikiText-2 test split | Standard, deterministic |
| Samples | 128 sequences, 2048 tokens each | ~262K tokens total |
| Metric | Token-level perplexity | Lower = better |

### Task Quality (Generation) — Directional Only

Fixed prompt sets with deterministic scoring. These are **directional signals**, not hard promotion gates — sample sizes are too small for statistical significance. The hard gate is perplexity on WikiText-2 above.

| Task | Dataset | Size | Metric | Scoring |
|------|---------|------|--------|---------|
| Prose | Fixed prompts in `scripts/eval_prompts/prose.jsonl` | 10 prompts | PPL of gold continuations | Auto (PPL) |
| Code | HumanEval subset in `scripts/eval_prompts/humaneval.jsonl` | 50 problems | pass@1 | Auto (execution) |
| Math | GSM8K subset in `scripts/eval_prompts/gsm8k.jsonl` | 100 problems | Exact match on final answer | Auto (regex) |
| Factual QA | TriviaQA subset in `scripts/eval_prompts/triviaqa.jsonl` | 100 questions | Exact match | Auto (string match) |

**Reproducibility**: All eval prompt files must be committed to the repo before Phase 3. Each file includes: dataset source, split, sample selection rule (e.g., "first N from test split"), tokenizer revision, and prompt template. Dataset metadata is logged in `experiments.jsonl` alongside results.

### Common Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Small model | Qwen2.5-7B | For Phase 1-2 iteration |
| Target model | Qwen2.5-14B | For Phase 3 benchmarks |
| Stretch model | Qwen2.5-72B | Phase 4 only (same family) |
| Context length | 2048 tokens | Matches existing harness |
| Sampling | Greedy (temperature=0) | Deterministic |
| Repetitions | 3 warm runs | Averaged |

## Success Criteria

### Phase 0 (hard gate — must pass to proceed)
1. **Mixed-quant loading works**: At least one toolchain path (A/B/C) produces a model with different bit-widths across layers that loads and generates correctly
2. **Memory accounting complete**: Tensor-by-tensor budget computed for 7B, 14B, and 72B from actual configs

### Phase 1-2 (minimum to continue)
1. **U-shape confirmed on 7B**: First/last 20% of layers show >2x sensitivity vs middle layers
2. **Allocation toolchain works**: Either automated (OptiQ) or manual (greedy knapsack) allocation produces valid mixed-precision models
3. **14B spot check** (before Phase 4 only): Profile first 3, middle 3, last 3 layers of 14B to verify U-shape transfers

### Phase 3 (experiment success)
1. **Perplexity improvement**: LOD allocation achieves >5% better perplexity than uniform at same memory budget (WikiText-2, hard gate)
2. **Memory savings**: At same perplexity as uniform Q4, LOD uses >20% less memory
3. **No throughput penalty**: tok/s within 5% of uniform quantization
4. **Task quality directional**: No task regresses by >10% vs uniform (directional, not hard gate — report confidence intervals)

### Phase 4 (stretch goal)
1. **70B fits**: Strict fit formula closes; model loads and generates within 24GB (peak Metal memory + RSS tracked)
2. **Quality acceptable**: Perplexity better than uniform Q2 70B (same family)
3. **Usable speed**: >1 tok/s decode
4. **Memory stable**: `psutil.virtual_memory().available` stays above 1GB throughout generation

## Structured Logging

All scripts emit structured JSON to `experiments.jsonl` using `scripts/experiment_utils.py`:
- `experiment_name`: e.g., `"layer_sensitivity_qwen7b_q2"`, `"optiq_qwen7b_2.5bpw"`, `"lod_bench_qwen14b_ushape"`
- `phase`: `"layer_sensitivity"`, `"optiq_allocation"`, `"lod_benchmark"`, `"large_model_deploy"`
- `config`: model, allocation strategy, target bit-width, etc.
- `results`: perplexity, tok/s, peak RSS, peak Metal memory (MB), available memory floor (GB), memory savings, etc.
- `env`: hardware info, OS version, available memory

## Rollback

Each phase is independent:
- Phase 1 sensitivity data is valuable even if later phases fail
- Phase 2 validates mlx-optiq for the project regardless of LOD results
- Phase 3 benchmarks quantify the LOD approach — negative results are still publishable
- Phase 4 is explicitly a stretch goal

## Dependencies

- `mlx` (already installed)
- `mlx-lm` (already installed)
- `mlx-optiq` / `optiq` (availability TBD — Phase 0 validates)
- `psutil` (already installed)
- `datasets` (for WikiText-2 loading — to install if needed)
- Quantized model checkpoints from HuggingFace (Qwen2.5 family)
