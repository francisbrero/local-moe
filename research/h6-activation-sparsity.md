# H6: Activation Sparsity Prediction

**Status**: Untested
**Analogy**: Frustum culling in game engines / sparse matrix computation
**Bottleneck addressed**: Memory bandwidth + compute

## The Insight

In 3D game engines, "frustum culling" skips rendering objects outside the camera's view frustum. You don't compute pixels for things the player can't see. This eliminates 50-90% of rendering work in typical scenes.

Transformer FFN layers have a parallel property: **most neuron activations are near zero for any given input**. After the gating function (SiLU/SwiGLU), a large fraction of intermediate values are near-zero or exactly zero. The matmul `output = activation @ W_down` computes a full dot product, but most terms contribute almost nothing.

If we could predict which neurons will activate strongly **before** loading their weights, we could skip loading and computing the near-zero ones entirely.

## Hypothesis

> By predicting which FFN neurons will have significant activations and only loading/computing those rows, we can reduce effective model size and bandwidth requirements by 50-80% per layer.

## Mechanism

```
Standard FFN:
  gate = x @ W_gate        # [hidden] @ [hidden, intermediate] → [intermediate]
  up   = x @ W_up          # [hidden] @ [hidden, intermediate] → [intermediate]
  act  = SiLU(gate) * up   # [intermediate] — many values near zero
  out  = act @ W_down      # [intermediate] @ [intermediate, hidden] → [hidden]

Sparse FFN:
  1. PREDICT: Which neurons will be active?
     - Small predictor: x @ W_predict → top-K neuron indices
     - Or: threshold on W_gate's first few rows

  2. LOAD: Only fetch rows of W_gate, W_up for predicted active neurons
     - And corresponding columns of W_down
     - From SSD if offloaded, or skip memory read if in RAM

  3. COMPUTE: Matmul with sparse subset only
     - [hidden] @ [hidden, K] instead of [hidden, intermediate]
     - K << intermediate (e.g., 2048 vs 14336)
```

## Why This Might Work

Strong empirical evidence for FFN activation sparsity:

1. **SiLU/SwiGLU gating**: The gating function naturally zeros out ~50-70% of neurons
2. **PowerInfer (SJTU)**: Demonstrated that activation sparsity is predictable — same neurons tend to activate for similar inputs. Achieved 11x speedup on consumer GPUs.
3. **DejaVu (2023)**: Showed that activating only 5% of MLP neurons gives 95%+ of model quality for many inputs
4. **ReLU models**: Some architectures use ReLU which produces exact zeros. Even smoother activations (SiLU) have practical sparsity.
5. **MoE is coarse-grained sparsity**: MoE activates K out of N experts. This is fine-grained sparsity within a single expert/FFN.

## Expected Impact

- **Bandwidth**: Read 20-50% of FFN weights per layer instead of 100%
- **Memory**: Only active neuron weights need to be in RAM (rest on SSD)
- **Compute**: Proportional reduction in matmul operations
- **Composability**: Stacks with quantization — Q4 on 30% of neurons < Q4 on 100%

## Quantifying the Savings

For a 70B model, FFN weights are ~70% of total parameters:
- Full Q4 FFN weights: ~28GB
- At 30% activation rate: ~8.4GB effective
- Combined with Q2: ~4.2GB effective
- Plus attention weights (~12GB Q4): total ~16-20GB — **fits in 24GB**

## Prior Art (Research Findings)

**Extensively researched area**. Activation sparsity is real, predictable, and exploitable. The critical variable is activation function: ReLU models have 85-95% native sparsity; SiLU/SwiGLU models need modification.

### ReLU vs SiLU/SwiGLU — The Fundamental Tradeoff

| Activation | Native Sparsity | Skip Computation? | Notes |
|-----------|----------------|-------------------|-------|
| ReLU | 85-95% (OPT, GPT) | Yes — exact zeros | Older architectures |
| SiLU/SwiGLU | ~0% native | Only with threshold | Llama, Qwen, Mistral |
| dReLU (TurboSparse) | 85-90% after fine-tune | Yes — hard zeros | Requires fine-tuning |
| ReLU2 | ~70% | Yes | `max(0,x)^2`, nearly no perf loss |

**"Sparsing Law"**: ReLU models improve in sparsity with more training data; SiLU models show the opposite trend.

### PowerInfer (v1 and v2)

- **[PowerInfer (2023)](https://arxiv.org/abs/2312.12456)** | [GitHub](https://github.com/SJTU-IPADS/PowerInfer): Exploits power-law distribution in neuron activation frequency. "Hot neurons" (consistently active) preloaded to GPU; "cold neurons" stay on CPU/SSD. Small trained MLP predicts per-token/per-layer which cold neurons activate. **OPT-175B on RTX 4090: 13.20 tok/s avg, 29.08 peak. Up to 27.8x speedup** over llama.cpp under memory-constrained offload.
- **[PowerInfer-2 (2024)](https://arxiv.org/abs/2406.06282)**: Extended to smartphones. TurboSparse-Mixtral-47B (~90% sparsity, only ~3-4B params active from 47B): **11.68 tok/s on smartphone (19GB)**, 2.13 tok/s at 7GB. For 7B models: ~40% memory savings vs llama.cpp.

### Deja Vu — Contextual Sparsity

- **[Deja Vu (2023)](https://arxiv.org/abs/2310.17157)** | [GitHub](https://github.com/FMInference/DejaVu): Lightweight 2-layer MLP predictor per layer. **Key trick: lookahead prediction** — input to attention layer k predicts MLP sparsity for layer k, hiding predictor latency in the pipeline. OPT-175B: **~85% total sparsity, no accuracy drop at 75% sparsity, 2x speedup** vs FasterTransformer. Predictor overhead: 18.1% of compute time. Focused on ReLU models (OPT series).

### Training-Free Approaches (No Fine-Tuning Required)

- **[TEAL (ICLR 2025)](https://arxiv.org/abs/2408.14690)** | [GitHub](https://github.com/FasterDecoding/TEAL): Magnitude-based thresholding on hidden states. No training, no predictor, no architecture change. **Works on SiLU/SwiGLU models (Llama-2, Llama-3, Mistral). 40% sparsity: near-zero degradation. 50% sparsity: 1.53-1.80x wall-clock speedup.** Universally applicable but sparsity is lower than ReLUfied models.
- **[SparseInfer (DATE 2025)](https://arxiv.org/abs/2411.12692)**: Predicts activation by comparing **only sign bits** of inputs and weights. For ReLU, you only need sign of `W^T x` to predict if neuron fires. **Predictor overhead: negligible (<0.1% runtime)**. Orders of magnitude cheaper than MLP predictors.
- **[ActTail (2025)](https://arxiv.org/abs/2603.12272)**: Non-uniform sparsity allocation using Heavy-Tailed Self-Regularization theory. **At 80% sparsity: 21.8% better perplexity than TEAL** on LLaMA-2-7B, 40.1% on LLaMA-2-13B. Different layers tolerate different sparsity levels.

### Training Sparsity Into Models

- **[TurboSparse (2024)](https://arxiv.org/abs/2406.05955)**: Replaced SiLU with novel `dReLU` via continued training. **TurboSparse-Mistral-7B: 90% neuron inactivity per layer. TurboSparse-Mixtral-47B: 85% per expert FFN.** Decoding speedup: 2-5x. On PowerInfer-2: 11 tok/s on mobile phone.
- **[ProSparse (2024)](https://arxiv.org/abs/2402.13516)**: ReLU + progressive sparsity regularization with sine-curve schedule. **LLaMA2-7B: 89.32% sparsity, LLaMA2-13B: 88.80%.** Performance comparable to original SwiGLU versions.
- **[ReLU Strikes Back (ICLR 2024)](https://openreview.net/forum?id=osoWxY8q2E)**: Simple ReLU substitution on LLaMA achieves ~67% sparsity without fine-tuning.
- **[ReLU2 (2024)](https://arxiv.org/abs/2402.03804)**: `max(0,x)^2` achieves 70% sparsity with nearly no performance loss at 1.3B scale.

### Dense-to-MoE Conversion (Coarser Granularity)

- **[CMoE (2025)](https://arxiv.org/abs/2502.04416)**: Training-free dense→MoE conversion. Analyzes activation rates to split neurons into shared (always-active) and routed (dynamic) experts. **Conversion under 5 minutes. 1.4-1.6x end-to-end speedup.**
- **[ExpertWeaver (2026)](https://arxiv.org/abs/2602.15521)**: Exploits GLU gating patterns. Identifies "universal neurons" vs "specialized neurons". Training-free, outperforms CMoE, FLAP, LLM-Pruner.

### Batch Size Matters

- **[Polar Sparsity (NeurIPS 2025)](https://arxiv.org/abs/2505.14884)**: At large batch sizes, union of active MLP neurons approaches full density, eliminating gains. **Attention head sparsity is stable across batch sizes while MLP sparsity is not.** For single-user inference (batch=1, our M4 case): MLP sparsity works perfectly. Up to 2.2x speedup.

### Apple Silicon / Metal Hardware

- **No native sparse GEMM** for neural network patterns on Apple Silicon. M4 uses ARM SME with 512-bit registers, 16x16 MAC arrays — dense matmul accelerator.
- **Metal has no cuSPARSE equivalent**. Sparse texture/buffer support exists (memory allocation, not compute skipping).
- **Implementation must be custom Metal kernel**: gather active neuron rows → dense matmul on subset. At batch=1, this is purely memory-bandwidth-bound — skipping inactive weight rows saves time proportional to sparsity ratio.
- A 90%-sparse model loading only 10% of FFN weight bytes = 10x reduction in FFN bandwidth = direct throughput gain at batch=1.

### Key Numbers

| Method | Sparsity | Speedup | Quality Impact | Training? |
|--------|----------|---------|---------------|-----------|
| PowerInfer (OPT-175B) | Model-dependent | 13-29 tok/s | Negligible | No (profiling only) |
| Deja Vu (OPT-175B) | 85% | 2x | None at 75% | Yes (predictor) |
| TEAL (Llama/Mistral) | 40-50% | 1.53-1.80x | Near-zero at 40% | No |
| TurboSparse (Mistral-7B) | 90% | 2-5x | Maintained | Yes (fine-tune) |
| ProSparse (LLaMA2-7B) | 89.32% | — | Comparable | Yes (fine-tune) |
| CMoE | Structural | 1.4-1.6x | Minimal | No (5 min) |
| SparseInfer | ReLU only | ~20% over SOTA | Within 1% | No |

### Recommended Path for M4

1. **Immediate (no training)**: Apply **TEAL thresholding** to existing Llama/Qwen checkpoints. 40-50% sparsity, 1.5-1.8x speedup. Custom Metal kernel for gather-based sparse GEMV.
2. **Quick conversion (5 min)**: **CMoE/ExpertWeaver** dense-to-MoE conversion on existing checkpoints. 1.4-1.6x speedup.
3. **If we control model choice**: Use **TurboSparse variants** (already available on HuggingFace). 85-90% sparsity with PowerInfer-style hot/cold placement.
4. **Optimal predictor**: **SparseInfer sign-bit prediction** for ReLUfied models — negligible overhead, very accurate.

## Open Questions

1. ~~How accurate does the predictor need to be?~~ DejaVu: 93-99% across layers is achievable and sufficient.
2. ~~What's the right predictor architecture?~~ **SparseInfer (sign-bit)** for ReLU models (negligible overhead), **TEAL (magnitude threshold)** for SiLU models (no predictor needed).
3. ~~What's the overhead of prediction vs savings?~~ DejaVu: 18% overhead. SparseInfer: <0.1% overhead. TEAL: zero overhead (just thresholding).
4. ~~Does this work with attention?~~ **Yes** — Polar Sparsity shows attention head sparsity is exploitable and stable across batch sizes.
5. Can we combine TEAL thresholding with LOD quantization (H5) — variable sparsity + variable precision by layer?
6. What's the actual Metal kernel speedup for gather-based sparse GEMV vs dense on M4? Need to benchmark the irregular access pattern overhead.

## Experiment Plan

### Phase 1: Activation analysis
- Run a 7B model and record FFN activations for diverse prompts
- Measure: sparsity ratio, distribution of activation magnitudes, which neurons are "always on" vs "input-dependent"
- Plot: activation heatmaps by layer and neuron

### Phase 2: Predictability
- For each layer, train a simple predictor (linear or small MLP) on (input hidden state → active neuron mask)
- Measure: prediction accuracy (precision, recall) at various sparsity thresholds
- Determine: can we predict top-K active neurons reliably?

### Phase 3: Quality at various sparsity levels
- Force-sparsify FFN layers to 50%, 30%, 20%, 10% active neurons
- Measure: perplexity degradation at each level
- Test: predicted sparsity vs random sparsity vs oracle (true top-K)

### Phase 4: Sparse inference prototype
- Implement sparse matmul in Metal (gather active rows → dense matmul on subset)
- Measure: actual speedup on M4 GPU, accounting for gather overhead
- Compare: sparse Q4 vs dense Q2 (which gives better quality per byte?)

## Risks

- Metal may not efficiently handle the irregular memory access patterns of sparse computation
- Prediction overhead could eat into speedup, especially for small models
- Activation sparsity patterns may be model-specific — what works for Llama may not work for Qwen
- Need to train predictors per model (not a one-time effort)
- Block-sparse may be needed for GPU efficiency, but reduces theoretical sparsity benefit
