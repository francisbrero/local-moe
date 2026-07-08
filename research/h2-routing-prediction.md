# H2: Expert Routing Prediction + Prefetch

**Status**: Tested — negative result on Qwen3-30B-A3B (model-dependent)
**Analogy**: CPU branch prediction / instruction prefetch
**Bottleneck addressed**: Memory capacity (SSD latency hiding)

## Experimental Result (H2, April 2026)

**Prediction viability is model-dependent.** Tested on Qwen3-30B-A3B (4-bit, 48 MoE layers, 128 experts, top-K=8):

| Metric | Result | Gate | Status |
|--------|--------|------|--------|
| Previous-layer recall | 6.2% | descriptive | Far below 78.8% prior art |
| Cross-layer Jaccard (L=2) | 0.032 | descriptive | Very low correlation |
| L=2 recall@8 (trained predictor) | 62.5% | ≥90% | **FAIL** |
| Oracle prefetch hit rate | 95.8% | ≥85% | PASS |
| Predicted prefetch hit rate | 5.6% | ≥85% | FAIL |
| Expert frequency Gini | 0.14 | — | Near-uniform (not Zipf) |

**Key finding**: Qwen3-30B-A3B has near-uniform expert routing (all 128 experts active, Gini=0.14) with very low cross-layer correlation. Prior art (ETH Zurich, Fate) achieving 93-97% recall was measured on models with concentrated/Zipf-like routing (DeepSeek V2, Phi-MoE). The prefetch *mechanism* works (oracle proves it), but routing for this specific model is too unpredictable.

**Recommendation**: Try models with concentrated routing patterns (e.g., DeepSeek V2 Lite). For Qwen3-30B-A3B, focus on other latency-hiding strategies.

See `dev/active/routing-prediction/context.md` for full results.

## The Insight

Modern CPUs don't wait for branch instructions to resolve before fetching the next instruction. Branch predictors guess which way a branch will go and speculatively prefetch instructions along the predicted path. When the prediction is right (~95% of the time), the pipeline never stalls.

MoE models have the same pattern: a router at each layer decides which K experts (out of 64) to activate. That decision takes one forward pass through a small gating network. By the time we know which experts are needed, it's too late to hide the SSD read latency.

## Hypothesis

> If we predict expert routing decisions 2-3 layers ahead using a lightweight predictor, we can prefetch expert weights from SSD while the GPU computes current layers, completely hiding SSD latency.

## Mechanism

```
Layer N is computing on GPU:
  1. Lightweight predictor estimates which experts Layer N+2 will need
     (based on hidden states at Layer N or N-1)
  2. Prefetch those experts from SSD → RAM in parallel with GPU compute
  3. By the time Layer N+2 starts, experts are already in RAM

If prediction is wrong:
  - Fall back to on-demand load (current behavior)
  - Or serve a low-precision fallback (HOBBIT-style)
```

## Why This Might Work

Expert routing has strong temporal patterns **in some models**:
- The same expert often activates for consecutive tokens
- Expert activation correlates across nearby layers
- Certain expert combinations co-occur frequently

Flash MOE observed ~71% OS page cache hit rate, meaning experts are reused. A smart predictor could push this much higher by prefetching before the miss happens.

> **H2 caveat**: These patterns were NOT observed in Qwen3-30B-A3B, which has near-uniform expert usage (temporal locality only 40%, cross-layer Jaccard only 3%). This assumption must be validated per model.

## Expected Impact

- **Latency hiding**: Overlap 5-7 GB/s SSD reads with GPU compute
- **Effective throughput**: Close to "all experts in RAM" speed for well-predicted workloads
- **Flash MOE's key insight was serial GPU/SSD**: This hypothesis challenges that by making prefetch non-blocking via separate threads

## Prior Art (Research Findings)

**This is a very active area** with 12+ serious papers from 2024-2026. Key finding: expert routing is highly predictable, and multiple approaches achieve 93-99% prediction accuracy.

### Cross-Layer Expert Prediction

- **[Pre-Attention Expert Prediction (ETH Zurich, 2025)](https://arxiv.org/abs/2511.10676)**: Predicts experts for layer N from *pre-attention* activations within the same layer (not prior layer output). Just 2 linear layers. **93-97% accuracy** across DeepSeek V2 Lite, Qwen3-30B, Phi-mini-MoE. ~15pp improvement over prior art. Solves the first-layer cold-start problem.
- **[Fate (2026)](https://arxiv.org/abs/2502.12224)**: Uses cross-layer gate inputs. **97.15% prefetch accuracy**, 99% cache hit rate. Up to 4.5x prefill speedup, 4.1x decode speedup. Previous-layer-only contribution: 78.8% accuracy (the baseline for simple approaches).
- **[Pre-Gated MoE (Microsoft, ISCA 2024)](https://arxiv.org/abs/2308.12066)**: Adds a pre-gating function one layer ahead. Algorithm-system co-design. Overlaps CPU-to-GPU expert migration with compute.

### Learned Predictors (Trained on Activation Traces)

- **[ProMoE (2024)](https://arxiv.org/abs/2410.22134)**: Trained learned predictor + stride prefetching + chunked prefetching. **2.2x avg speedup** (up to 3.21x) in prefill, **2.07x avg** (up to 5.02x) in decode.
- **[MoE-Beyond (2025)](https://arxiv.org/abs/2508.17137)**: Lightweight transformer trained on 66M activation traces. **Cache hit rate: 17% → 72%** at only 10% expert cache capacity. 97.5% prediction accuracy on unseen prompts. Most dramatic improvement reported.
- **[ExpertFlow v1 (2024)](https://arxiv.org/abs/2410.17954)**: Transformer-based predictor predicts *all* experts for entire forward pass at once. Up to 93.72% GPU memory savings, 2-10x speedup.
- **[ExpertFlow v2 (2025)](https://arxiv.org/abs/2510.26730)**: Adaptive prefetch horizon (not fixed lookahead) based on runtime stats. Model stall time reduced to <0.1% of baseline.

### Shadow Model Approach

- **[OD-MoE (2025)](https://arxiv.org/abs/2512.03927)**: Runs a quantized "shadow" copy of the model in parallel. Shadow runs faster (quantized) and predicts expert activations multiple layers ahead. **FP16 shadow: 99.94% accuracy**, INT8: 97.34%, NF4: 95.67%. Delivers ~75% of fully-GPU-cached speed at 1/3 GPU memory. **Risk**: doubles model memory footprint — may be impractical on 16GB.

### Speculative Decoding + Expert Prefetch Combined

- **[MoE-SpeQ (2025)](https://arxiv.org/abs/2511.14102)**: Uses draft model to *also* predict which experts the full model will need. 4-bit quantized draft achieves >90% prediction accuracy. **Up to 3.3x speedup** on Qwen models. If you're already doing speculative decoding, expert predictions are nearly free.

### Expert Substitution (Handling Prediction Misses)

- **[BuddyMoE (2025)](https://arxiv.org/abs/2511.10054)**: When prefetch fails, substitute a functionally similar "buddy" expert already in memory. Buddy sets built offline using co-activation frequency. Eliminates stalls on misses. **Complementary to any predictor** — prediction + substitution > prediction alone.

### System-Level Scheduling

- **[PreScope (2025)](https://arxiv.org/abs/2509.23638)**: Layer-Aware Predictor + Cross-Layer Scheduling + AsyncIO optimizer. **141% higher throughput**, 74.6% lower latency. Key insight: layer-specific (not generic) predictors + global scheduling across layers.
- **[DuoServe-MoE (2025)](https://arxiv.org/abs/2509.07379)**: Separates prefill (dense) and decode (sparse) with tailored strategies. **1.42-7.54x latency improvement**. Peak memory only 15% of full model.

### Cache Replacement

- **[SpecMD (2026)](https://arxiv.org/abs/2602.03921)**: Benchmarking study revealing MoE expert access does NOT follow temporal locality (LRU/LFU assumptions fail). Proposes "Least-Stale" eviction. **85x fewer collision misses** vs LRU, 88%+ hit rates at 5% VRAM.

### CPU Branch Prediction Analogy

- **[Speculating Experts (2026)](https://arxiv.org/abs/2603.19289)**: Explicitly draws the branch predictor analogy. Key difference: branch misprediction costs ~15 pipeline cycles; expert load misprediction costs ~10ms of SSD transfer. Stakes are enormously higher — need 97%+ accuracy.

### Key Numbers

| Method | Prediction Accuracy | Speedup | Complexity |
|--------|-------------------|---------|------------|
| Previous-layer-only (baseline) | 78.8% | — | Negligible |
| Pre-attention predictor (ETH) | 93-97% | — | 2 linear layers |
| Fate (cross-layer gate) | 97.15% | 4.1-4.5x | Low |
| MoE-Beyond (trained transformer) | 97.5% | 17%→72% hit rate | Medium |
| ProMoE (learned predictor) | — | 2.07-5.02x | Medium |
| OD-MoE (NF4 shadow) | 95.67% | ~75% of full-cache | High (memory) |
| PreScope (layer-aware + scheduling) | — | 141% throughput | Medium |

### Recommended Approach for M4

Based on the literature, the most practical combination:
1. **Cross-layer linear predictor** (Fate/ETH pre-attention) — run on Metal, minimal overhead
2. **Async SSD reads** overlapping with GPU compute (PreScope's AsyncIO approach)
3. **Co-activation table** for expert substitution on miss (BuddyMoE — trivial to implement)
4. **Least-Stale eviction** instead of LRU (SpecMD finding)

## Open Questions

1. ~~How accurately can we predict routing 2-3 layers ahead?~~ **Answered (H2): model-dependent.** Prior art reports 93-97% on DeepSeek V2/Phi-MoE, but Qwen3-30B-A3B only achieves 62.5% recall@8 at L=2 due to near-uniform routing. The predictor architecture (2 linear layers) is sound — the model's routing pattern is the bottleneck.
2. Does Apple Silicon's unified memory controller allow true concurrent SSD DMA + GPU compute? Flash MOE says concurrent access hurts by 73%. But **prefetch to RAM staging buffer** might avoid the contention — PreScope's AsyncIO suggests this works. **H0 Phase 4a confirmed**: GPU/SSD contention is only 0.2% on M4 Pro (negligible).
3. ~~What's the right predictor architecture?~~ **Answered: 2-layer linear predictor on pre-attention activations** (ETH Zurich) is the sweet spot of accuracy vs overhead. **H2 confirmed**: predictor latency is only 0.189ms/token — negligible.
4. Can we combine BuddyMoE substitution with HOBBIT mixed-precision fallback for a two-tier miss handling strategy?
5. Is the SpecMD "Least-Stale" eviction practical on NVMe (designed for GPU VRAM)?
6. **NEW (H2)**: Which MoE models have concentrated vs uniform routing? This determines whether prediction-based prefetching is viable. Need to profile routing patterns across model families before committing to this approach.

## Experiment Plan

### Phase 1: Routing pattern analysis
- Run an MoE model and log expert activations per layer per token
- Measure: cross-layer correlation, temporal locality, expert co-occurrence
- Answer: how predictable are routing decisions?

### Phase 2: Predictor design
- Train a small MLP or lookup table on routing traces
- Measure: prediction accuracy at 1, 2, 3 layer lookahead
- Determine: minimum accuracy needed for net speedup

### Phase 3: Prefetch pipeline
- Implement async SSD prefetch on a background thread
- Test RAM staging buffer approach to avoid GPU memory contention
- Measure: end-to-end tok/s with prediction + prefetch vs baseline

## Risks

- ~~Prediction accuracy may be too low, wasting SSD bandwidth on wrong experts~~ **Confirmed (H2)**: Qwen3-30B-A3B routing is too uniform for accurate prediction. Model-dependent risk is real.
- ~~Unified memory contention may negate prefetch benefits (Flash MOE's finding)~~ **Mitigated (H0)**: GPU/SSD contention is only 0.2% on M4 Pro.
- ~~Predictor overhead (even small MLP) adds latency per layer~~ **Mitigated (H2)**: 0.189ms/token is negligible.
- RAM staging buffer reduces effective memory for other uses
- **NEW**: Prior art numbers (93-97% recall) may only apply to models with Zipf-like routing. Must profile routing patterns per model before committing to this approach.
