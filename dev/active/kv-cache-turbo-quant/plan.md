# Experiment: KV Cache Compression (TurboQuant)

**Issue**: #3

## Hypothesis

On a 16GB M4 MacBook Pro, we can compress the KV cache to 3-4 bits per element using TurboQuant-style random rotation + scalar quantization, achieving 3-4x memory reduction with negligible accuracy loss. For the harness models at 8K-16K context, this saves 288-624 MB — enough to meaningfully extend context length or free headroom for larger models. At 32K+ contexts or with larger-KV models (8+ KV heads), savings exceed 1 GB. The technique is composable with expert offloading (Issue #2).

## Background

TurboQuant (Google, ICLR 2026) is a two-stage KV cache compression method:
1. **PolarQuant**: Random orthogonal rotation normalizes vector distributions, then Lloyd-Max optimal scalar quantization compresses per-coordinate
2. **QJL**: 1-bit residual correction via Quantized Johnson-Lindenstrauss transform

Key finding from the `tonbistudio/turboquant-pytorch` reference implementation: the QJL stage actually *hurts* generation quality in attention contexts because softmax exponentially amplifies quantization noise. Their V3 (MSE-only, no QJL) passed 18/18 generation tests vs 0/27 with QJL. We will implement MSE-only (no QJL).

A llama.cpp fork (`TheTom/turboquant_plus`) demonstrates Metal kernels achieving parity with q8_0 prefill speed and +33.9% decode speedup on M1 Max at long context. This validates the approach on Apple Silicon.

## Hardware Note

Development machine is a 24GB M4 Pro. Target is 16GB base M4. We use `create_memory_pressure()` to simulate the 16GB memory constraint for memory/OOM testing. However, memory pressure simulation **cannot replicate** the target machine's GPU/CPU performance characteristics, cache hierarchy, or unified-memory contention behavior. Therefore:
- **Memory and OOM tests**: Valid on dev machine with pressure simulation
- **Algorithm correctness and accuracy tests**: Valid on dev machine
- **Throughput/TTFT/latency tests**: Results are indicative only. Final performance validation must be run on the actual 16GB M4 target before declaring throughput success criteria met. Plan marks these as "pending target validation" in results.

## Approach

### Pre-Phase: Measure Loaded Memory Baselines

Before any benchmarking, measure actual post-load resident memory for each harness tier to replace the approximate budget table with ground truth:

1. For each tier (S, M, XL): load the model via MLX, then measure:
   - `psutil.virtual_memory().available` (available memory after load)
   - `get_peak_rss_mb()` (process RSS)
   - `mx.metal.get_peak_memory()` (Metal GPU memory)
2. Compute actual headroom: `measured_available - 500MB_safety_margin = KV budget`
3. Gate long-context tests: only run XL at 16K/32K if measured KV budget exceeds FP16 KV size for that context. If not, adjust context lengths or skip.
4. Log measurements to `experiments.jsonl` as `phase: "memory_baseline"`.
5. Update the per-workload budget table in this plan with measured values.

**Note on simulated 16GB**: All memory/OOM results using `create_memory_pressure()` on the 24GB dev machine are **provisional**. They validate behavior under memory pressure but do not prove viability on the actual 16GB M4 target, which has different absolute DRAM limits and unified-memory contention behavior. Any "this fits on 16GB" claim requires final validation on the target hardware.

### Phase 0: Validate Existing Implementations + Decision Gate (Day 1)

Before writing custom code, evaluate what's already available:

1. **Test `turboquant-pytorch` reference implementation**:
   - Install and run their validation suite against a small model (Qwen3-0.5B or similar)
   - Verify claimed accuracy metrics: >0.999 cosine similarity for K4/V4
   - Measure memory savings on our hardware
   - Understand the API surface for KV cache integration

2. **Test `turboquant_plus` llama.cpp fork**:
   - Build with Metal support on M4
   - Run with `-ctk turbo4 -ctv turbo4` flags
   - Benchmark against standard q8_0 KV cache

3. **Decision gate**:
   - If `turboquant_plus` works on M4 and validates the approach (accuracy, memory savings, reasonable throughput): **scope this issue to characterization and benchmarking only**. File a follow-up issue for custom MLX/Metal implementation if needed.
   - If existing implementations don't work on M4 (build failures, missing Metal support, accuracy problems): proceed to Phase 1 for custom implementation.
   - Document the decision and rationale in `context.md`.

### Phase 0b: MLX Integration Audit

Before implementing the cache wrapper, determine how compressed KV integrates with the MLX inference path:

1. **Inspect `mlx_lm` internals**: Check whether `mlx_lm.stream_generate()` exposes a cache hook or custom cache class. Inspect the attention implementation to understand how K/V are stored and accessed.
2. **Decide integration path**:
   - **Option A**: If `mlx_lm` supports pluggable cache (e.g., via a `cache_class` parameter or overrideable cache object), use it directly.
   - **Option B**: Write a custom decode loop that bypasses `stream_generate()` but reuses the model's forward pass, injecting compressed K/V at the attention layer.
   - **Option C**: If full integration requires forking/patching MLX-LM attention internals, scope this issue to **standalone attention microbenchmarks + characterization only** and file full model integration as a follow-up issue.
3. **Document decision** in `context.md` before proceeding to Phase 1.

### Phase 1: Core Algorithm Implementation in MLX (Days 2-3)

**Note**: Only proceed here if Phase 0 decision gate indicates custom implementation is needed.

**Scope control**: This issue delivers ONE of two outcomes:
- **Path A (existing tools work)**: Characterization and benchmarking of `turboquant_plus` and/or `turboquant-pytorch` on our hardware. No custom implementation.
- **Path B (custom implementation needed)**: Minimal K4/V4 prototype on a single model tier (M or S), validated for correctness and memory savings. Ablations (Phase 2b), cross-tier testing, and expert offloading interaction are filed as follow-up issues if the baseline proves the concept.

Implement a **minimal TurboQuant baseline** in MLX: fixed K4/V4, no residual window, no layer-adaptive precision. The integration path (A, B, or C from Phase 0b) determines whether the cache wrapper targets full model inference or standalone attention microbenchmarks.

1. **Lloyd-Max Scalar Quantizer** (`scripts/turbo_quant/lloyd_max.py`):
   - Precompute optimal codebooks for Gaussian distributions at 2, 3, 4, 6, 8 bits
   - Use iterative centroid optimization (converges in ~20 iterations)
   - Cache codebooks as constants — they're data-oblivious (same for any model)

2. **Random Orthogonal Rotation** (`scripts/turbo_quant/rotation.py`):
   - Generate rotation matrix via QR decomposition of random Gaussian matrix: `Q, _ = mx.linalg.qr(mx.random.normal(shape=(d, d)))`
   - One matrix per head dimension, generated once at init from a fixed seed
   - Store as MLX array for GPU-accelerated matrix multiply

3. **Quantize/Dequantize** (`scripts/turbo_quant/core.py`):
   - **Quantize (K and V)**: normalize → rotate → scalar quantize per-coordinate → store indices + norm
   - **Dequantize**: look up centroids from codebook (stays in rotated space) → restore norm
   - **Key design: operate in rotated space**. Both K and V are stored rotated and quantized. During attention:
     - Keys: queries are rotated once (`Q_rot = Q @ R`) and dot products are computed against rotated K directly. Since `Q @ R @ R^T @ K^T = Q @ K^T` (R is orthogonal), inner products are preserved exactly.
     - Values: dequantized V stays in rotated space; the weighted sum `softmax(scores) @ V_rotated` produces a rotated output, which is inverse-rotated once per token (`output @ R^T`). This is O(d^2) per token, not per cached position.
   - This avoids the O(seq_len * d^2) inverse rotation on every cached token during decode.
   - Support bit-packing for real memory savings (not just index arrays)

4. **KV Cache Wrapper** (`scripts/turbo_quant/cache.py`):
   - Drop-in replacement for standard KV cache in MLX inference
   - **Minimal baseline**: fixed K4/V4, both K and V rotated+quantized, all layers same precision, no residual window
   - Rotates Q at attention time (one matmul per head per layer) and inverse-rotates attention output (one matmul per head per layer) — both are O(d^2) per token, independent of sequence length
   - Reports direct byte accounting: `cache.kv_bytes()` returns exact storage used (packed indices + norms + metadata), separate from RSS/GPU peak metrics

### Phase 2: Accuracy Validation + Ablations (Day 3-4)

Validate the minimal baseline first, then test each heuristic as an isolated ablation.

**2a. Minimal baseline validation** (`scripts/turbo_quant/test_accuracy.py`):

1. **Attention fidelity test**:
   - Compare compressed vs FP16 attention output (cosine similarity, L2 error)
   - Target: >0.999 cosine similarity for K4/V4
   - Test across context lengths: 512, 2K, 4K, 8K, 16K
   - Explicitly track cosine similarity vs context length to detect error growth at longer contexts

2. **Perplexity test**:
   - Use `PERPLEXITY_PASSAGES` from existing `scripts/prepare.py` for short-context validation
   - Additionally, construct a long-context perplexity evaluation: concatenate passages to create 4K+ token sequences, measure chunked perplexity
   - Measure at 4-bit and 3-bit configurations
   - Target: within 0.5 PPL of FP16 baseline for 4-bit at all context lengths

3. **Generation quality test**:
   - Needle-in-haystack at 2K, 4K, 8K, and 16K context lengths (16K is the regime where KV compression matters most)
   - Compare generated text against FP16 baseline
   - Explicitly tie success criteria to context length: if accuracy degrades at 8K+ but passes at 4K, document the crossover point

**2b. Heuristic ablations** (each tested independently against the baseline):

4. **K-only rotation**: Rotate and quantize K, but quantize V without rotation (per-coordinate only). Tests whether rotation is needed for values or just keys.
5. **Residual FP16 window**: Keep last 128 tokens uncompressed. Measure PPL delta vs baseline.
6. **Asymmetric K/V precision**: K6/V4 and K4/V2 configurations. Measure accuracy vs compression tradeoff.
7. **Layer-adaptive precision**: First/last 2 layers at K6/V6, middle layers at K4/V4. Measure PPL impact.

Each ablation is logged separately to `experiments.jsonl` so improvements are attributable.

### Phase 3: Memory & Performance Benchmarking (Day 4-5)

1. **Memory savings measurement** (`scripts/turbo_quant/bench_memory.py`):
   - **Primary metric**: Direct byte accounting via `cache.kv_bytes()` — exact storage used for packed indices, norms, metadata. This is the ground truth for compression ratio, independent of RSS noise from model weights and allocator behavior.
   - **Supporting metrics**: Peak RSS and peak Metal/GPU memory (`mx.metal.get_peak_memory()`) to confirm system-level impact, but these are coarse signals on top of multi-GB model weights.
   - **Cache growth tracking**: Measure `cache.kv_bytes()` at fixed model load while incrementally adding tokens (100, 500, 1K, 2K, 4K, 8K, 16K) to validate linear growth and compute per-token storage cost.
   - Test at context lengths: 512, 2K, 4K, 8K, 16K tokens
   - Log to `experiments.jsonl`

2. **Throughput benchmark** (`scripts/turbo_quant/bench_throughput.py`):
   - Measure tok/s for decode with compressed vs FP16 KV cache
   - Measure TTFT (time to first token) impact
   - Benchmark query rotation latency overhead per token
   - Benchmark dequantization latency during attention (codebook lookup for V reconstruction)

3. **Stress test — memory pressure** (`scripts/turbo_quant/bench_stress.py`):

   **Primary stress test (XL at 16K)**: Repeatable, measurable pressure differential.
   - Use `create_memory_pressure()` to simulate 16GB constraint. Load XL (14B, 4-bit weights ~8.5GB).
   - At 16K: FP16 KV = 768 MB, leaving virtually no headroom for transient buffers. 4-bit KV = 192 MB, leaving ~1.1 GB headroom.
   - **Test pair**: Run XL at 16K with FP16 KV, then with 4-bit compressed KV.
   - Success defined by measured behavior: `psutil.virtual_memory().available` floor > 500 MB and pageout delta < 100 MB for compressed run.

   **Optional boundary probe (XL at 32K)**: Only if 16K test passes cleanly.
   - FP16 KV at 32K (1536 MB) exceeds the budget — expect clear failure. 4-bit KV (384 MB) is marginal after accounting for transient buffers.
   - This is an exploratory probe, not a required success case. Document observed behavior regardless of outcome.

4. **Interaction with model sizes**:
   - Test with S (0.5B), M (3B), XL (14B) model tiers
   - Determine largest model that fits with compressed KV at 8K+ context
   - Compare: model that fits with FP16 KV vs with 4-bit KV

### Phase 4: Architecture Interaction Testing (Day 5)

1. **GQA architecture validation**:
   - All models in our harness use GQA (Qwen2.5, Llama 3 family). Validate compression works correctly with GQA's shared KV heads.
   - Test that quantization noise doesn't interact badly with KV head sharing (one compressed KV head serves multiple query heads)
   - Measure per-head compression quality to verify uniformity

2. **Interaction with expert offloading** (if Issue #2 is complete):
   - Test compressed KV cache alongside expert streaming
   - Measure combined memory savings
   - Verify no interaction effects on accuracy

**Scope narrowing**: Issue #3 requests testing GQA, MQA, and MHA architectures. This experiment covers GQA only (all harness models are GQA). MHA and MQA testing is deferred because no MHA/MQA model is in the harness registry. A follow-up issue will be filed to add MHA/MQA attention microbenchmarks using synthetic attention tensors if GQA results are positive.

## Memory Budget Analysis

KV cache size formula: `2 (K+V) * num_layers * num_kv_heads * head_dim * seq_len * bytes_per_element`

### KV Cache Sizes (from harness model configs)

| Model (Harness Tier) | Layers | KV Heads | Head Dim | FP16 KV @ 2K | 4-bit KV @ 2K | Savings |
|----------------------|--------|----------|----------|--------------|----------------|---------|
| Qwen2.5-0.5B (S) | 24 | 2 | 64 | 12 MB | 3 MB | 9 MB |
| Qwen2.5-3B (M) | 36 | 2 | 128 | 36 MB | 9 MB | 27 MB |
| Qwen2.5-14B (XL) | 48 | 4 | 128 | 96 MB | 24 MB | 72 MB |

**Analysis-only reference models** (not in harness, for composability analysis with Issue #2):

| Model | Layers | KV Heads | Head Dim | FP16 KV @ 2K | 4-bit KV @ 2K | Savings |
|-------|--------|----------|----------|--------------|----------------|---------|
| Qwen3-30B-A3B (MoE) | 48 | 4 | 128 | 96 MB | 24 MB | 72 MB |
| Mixtral-8x7B | 32 | 8 | 128 | 128 MB | 32 MB | 96 MB |

At longer contexts (Qwen2.5-14B / XL tier):

| Context | FP16 KV | 4-bit KV | 3-bit KV | Savings |
|---------|---------|----------|----------|---------|
| 2K | 96 MB | 24 MB | 18 MB | 72-78 MB |
| 4K | 192 MB | 48 MB | 36 MB | 144-156 MB |
| 8K | 384 MB | 96 MB | 72 MB | 288-312 MB |
| 16K | 768 MB | 192 MB | 144 MB | 576-624 MB |
| 32K | 1536 MB | 384 MB | 288 MB | 1.1-1.2 GB |

### Per-Workload Resident Memory Budget (estimated — will be replaced by Pre-Phase measurements)

Target machine: 16GB M4 MacBook Pro (~11GB usable). Note: development machine is 24GB M4 Pro — use `create_memory_pressure()` to simulate 16GB constraint. **These are estimates from `approx_size_gb`; Pre-Phase will measure actual loaded residency and update this table.**

| Component | S (0.5B) | M (3B) | XL (14B) |
|-----------|----------|--------|----------|
| OS + apps headroom | 5 GB | 5 GB | 5 GB |
| Model weights (4-bit) | 0.4 GB | 1.8 GB | 8.5 GB |
| Python/MLX runtime | 0.5 GB | 0.5 GB | 0.5 GB |
| Metal scratch buffers | 0.2 GB | 0.3 GB | 0.5 GB |
| **Subtotal (fixed)** | **6.1 GB** | **7.6 GB** | **14.5 GB** |
| **Remaining for KV** | **9.9 GB** | **8.4 GB** | **1.5 GB** |

XL at 16K context: FP16 KV = 768 MB (fits, but only 732 MB headroom left — no room for transient buffers). At 32K: FP16 KV = 1536 MB — **exceeds budget by ~36 MB**.

**Important**: The remaining headroom for KV does NOT account for transient quantization/dequantization buffers (rotation matrix: d*d*2 bytes per head ≈ 32KB, codebook: negligible, norm storage: seq_len*4 bytes per head) or allocator fragmentation. For 4-bit KV, add ~10% overhead for norms and metadata. The budget table represents raw KV storage only.

**Benchmark validity by context length (XL tier, simulated 16GB)**:

| Context | FP16 KV fits? | 4-bit KV fits? | Test type |
|---------|---------------|----------------|-----------|
| 2K-8K | Yes | Yes | Normal benchmark |
| 16K | Marginal (no headroom for transient buffers) | Yes (1.1 GB headroom) | Stress test |
| 32K | **No** (over budget) | Marginal (needs validation) | **OOM boundary test only** |

All stress tests require `psutil.virtual_memory().available` floor monitoring and `vm_stat` pageout tracking. A configuration is only "viable" if it completes with available memory floor > 500 MB and pageout delta < 100 MB.

## Canonical Evaluation Workload

| Parameter | Value | Notes |
|-----------|-------|-------|
| Models | Qwen2.5-0.5B (S), Qwen2.5-3B (M), Qwen2.5-14B (XL) | Harness tiers; XL for stress test |
| Bit widths | 4-bit, 3-bit | Primary configurations |
| Context lengths | 512, 2K, 4K, 8K, 16K | Scaling behavior; 16K for stress test |
| Prompt | `BENCH_PROMPT` from `scripts/prepare.py` | Consistent with Issue #2 |
| Max decode tokens | 256 | |
| Sampling | Greedy (temperature=0) | Deterministic |
| Repetitions | 3 (warm runs) | Steady-state |
| Perplexity corpus | `PERPLEXITY_PASSAGES` from `scripts/prepare.py` | 4 passages |
| Time budget | 300s per benchmark | Abort on timeout |
| Residual window | 0 and 128 tokens | Test both configurations |

## Structured Logging

All scripts emit to `experiments.jsonl` using `experiment_utils.log_experiment()`:
- `experiment_name`: e.g., `"turbo_quant_accuracy_k4v4_2k"`, `"turbo_quant_memory_llama8b_4k"`
- `phase`: `"turbo_quant_validation"`, `"turbo_quant_accuracy"`, `"turbo_quant_memory"`, `"turbo_quant_throughput"`
- `config`: bit width, model, context length, residual window size, layer adaptive config
- `results`: cosine similarity, PPL, peak RSS, tok/s, TTFT, compression ratio
- `env`: hardware info via `experiment_utils.get_environment_info()`

## Metrics

All planned:
- [ ] Cosine similarity of attention output: compressed vs FP16
- [ ] L2 error of attention output: compressed vs FP16
- [ ] Perplexity delta vs FP16 baseline (target: <0.5 PPL increase)
- [ ] Peak RSS (MB) with FP16 vs 4-bit vs 3-bit KV cache
- [ ] Peak Metal/GPU memory via `mx.metal.get_peak_memory()`
- [ ] KV cache direct byte accounting via `cache.kv_bytes()` (packed indices + norms + metadata)
- [ ] Per-token KV storage cost (bytes/token) from cache growth measurement
- [ ] Compression ratio: `cache.kv_bytes()` vs equivalent FP16 storage, including all overhead
- [ ] Quantization latency (ms) per token: rotation + quantize (K and V)
- [ ] Query rotation latency (ms) per token: single matmul per head
- [ ] Dequantization latency (ms) during attention: codebook lookup for V reconstruction
- [ ] tok/s (decode) with compressed KV cache
- [ ] TTFT (time to first token) impact
- [ ] Generation quality: needle-in-haystack pass/fail
- [ ] `psutil.virtual_memory().available` floor during runs
- [ ] Pageout delta via `vm_stat` during benchmarks

**Note on cache hit rate**: This standard project metric (from CLAUDE.md) is not applicable to KV cache compression — there is no expert or layer cache involved. The equivalent metrics for this experiment are compression ratio and pageout delta, which measure how effectively the compression reduces memory pressure.

## Success Criteria

### Minimum viability (compression works)
1. **Accuracy**: >0.999 cosine similarity for K4/V4 attention output
2. **Perplexity**: <0.5 PPL increase vs FP16 at 4-bit
3. **Memory**: `cache.kv_bytes()` shows >2x compression vs FP16 equivalent at 4K+ context
4. **Overhead**: Quantization latency <5ms per token

### Full success (practical for inference)
1. **Accuracy**: >0.999 cosine at K4/V4, >0.995 at K3/V3
2. **Perplexity**: <0.2 PPL increase at 4-bit
3. **Memory**: >3x compression ratio via `cache.kv_bytes()` (including all overhead), confirmed by RSS delta at 8K+ context
4. **Throughput** (gated by implementation path):
   - **If using `turboquant_plus` or custom Metal kernels**: tok/s within 90% of FP16 baseline
   - **If using MLX Python prototype**: throughput is informational only — success is correctness + compression ratio + bounded overhead (<5ms quantize latency). Throughput parity deferred to Metal-focused follow-up.
   - *All throughput results pending final validation on 16GB M4 target — see Hardware Note.*
5. **Generation**: Needle-in-haystack passes at 4K with K4/V4
6. **Composability**: Validated alongside model weight quantization (no compounding accuracy loss)

### Stretch goals
1. **3-bit compression**: K3/V3 with <1.0 PPL increase
2. **Metal kernel**: Custom Metal dequant kernel for attention (faster than MLX generic path)
3. **Integration with Issue #2**: Compressed KV + expert offloading measured together

## File Structure

```
scripts/turbo_quant/
├── __init__.py
├── lloyd_max.py          # Lloyd-Max optimal scalar quantizer
├── rotation.py           # Random orthogonal rotation via QR
├── core.py               # Quantize/dequantize functions
├── cache.py              # KV cache wrapper for MLX inference (with kv_bytes() accounting)
├── test_accuracy.py      # Attention fidelity + perplexity + needle tests
├── bench_memory.py       # Memory savings benchmark (direct byte accounting + RSS)
├── bench_throughput.py   # Throughput benchmark
├── bench_stress.py       # OOM boundary / memory pressure stress test
└── test_existing.py      # Phase 0: test existing implementations
```

## Baselines

FP16 KV cache baseline using the existing harness:

```bash
# Prepare models (download if needed)
uv run python scripts/prepare.py --tier S
uv run python scripts/prepare.py --tier M
uv run python scripts/prepare.py --tier XL

# Run FP16 baseline for each tier
# Edit scripts/benchmark.py: MODEL_TIER="S", EXPERIMENT_NAME="kv_baseline_fp16_S"
uv run python scripts/run.py

# Compressed KV benchmark (after Phase 1 implementation)
uv run python scripts/turbo_quant/bench_memory.py --tier S --bits 4
uv run python scripts/turbo_quant/bench_throughput.py --tier S --bits 4
uv run python scripts/turbo_quant/bench_stress.py --tier XL --bits 4 --context 32768
```

## Rollback

Each phase produces standalone results:
- Phase 0 benchmarks existing implementations (useful even if we don't write custom code)
- Core algorithm is pure math — validated independently before integration
- Benchmarks are standalone scripts reusable for other KV cache experiments
- If TurboQuant doesn't work, results inform other compression approaches (per-channel quantization, sliding window, etc.)

## References

- [TurboQuant paper (arXiv:2504.19874)](https://arxiv.org/abs/2504.19874)
- [PolarQuant paper (arXiv:2502.02617)](https://arxiv.org/abs/2502.02617)
- [QJL paper (arXiv:2406.03482)](https://arxiv.org/abs/2406.03482)
- [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) — Reference PyTorch implementation
- [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) — llama.cpp fork with Metal kernels
- [Google Research blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
