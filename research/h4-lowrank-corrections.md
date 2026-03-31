# H4: Low-Rank Base + Sparse SSD Corrections

**Status**: Untested
**Analogy**: Compressed sensing / progressive image rendering
**Bottleneck addressed**: Memory capacity

## The Insight

Compressed sensing in MRI reconstructs full images from far fewer measurements than the Nyquist rate by exploiting the fact that natural images are sparse in some basis. You don't need all the data — you need the right projection plus a way to fill in the gaps.

LoRA proved that the difference between a pre-trained model and a fine-tuned model lives in a very low-rank subspace — a few matrices of rank 16-64 capture most of the adaptation. This implies that **weight matrices themselves have significant redundant structure**.

Progressive JPEG works similarly: send a coarse version fast, then refine it. The user sees something useful immediately.

## Hypothesis

> Decompose each weight matrix into a low-rank base (fits in RAM) plus sparse corrections (stored on SSD). The base provides a usable approximation for most tokens. Corrections are loaded on-demand only for layers/tokens where the approximation is insufficient.

## Mechanism

```
Offline (one-time preparation):
  For each weight matrix W (shape [m, n]):
    1. Compute SVD: W = U @ S @ V^T
    2. Keep top-k singular values as BASE = U[:,:k] @ S[:k,:k] @ V[:,:k]^T
    3. Compute CORRECTION = W - BASE (sparse, most entries near zero)
    4. Quantize and compress CORRECTION, store on SSD
    5. Store BASE in RAM (much smaller: m*k + k*n vs m*n)

Inference:
  1. Run forward pass using BASE weights (fast, all in RAM)
  2. Monitor output quality signal (entropy, confidence)
  3. If quality is degraded on certain layers:
     a. Load CORRECTION for those layers from SSD
     b. Re-compute: output = x @ (BASE + CORRECTION)
     c. Continue with corrected activations
```

## Memory Math

For a 70B model with typical layer dimensions:
- Full Q4 weights: ~40GB
- Rank-256 SVD base (Q4): ~8-12GB (5-8x reduction depending on architecture)
- Corrections (sparse, quantized): ~30GB on SSD

**This could fit a 70B model's working approximation in 24GB RAM.**

## Why This Might Work

- Weight matrices in large LLMs are known to have rapidly decaying singular value spectra
- LoRA (rank 16-64) captures fine-tuning deltas — the base model's structure is even more compressible
- The correction is sparse (many near-zero entries) — compresses well on SSD
- Different layers have different effective ranks — some are nearly low-rank already

## Expected Impact

- **Memory**: 70B model base fits in ~10-12GB RAM
- **Quality**: Rank-256 base retains ~90-95% of model quality (hypothesis — needs testing)
- **Speed**: Most inference runs at RAM speed; SSD corrections only for critical layers
- **Composability**: Combine with KV cache compression (TurboQuant) for full 70B on 24GB

## Prior Art (Research Findings)

This is a very active area. The W = LR + S decomposition is well-validated, but the **end-to-end system** (LR in RAM, S streamed from SSD on demand) has **not been built**.

### SVD-Based LLM Compression

- **[SVD-LLM (ICLR 2025)](https://arxiv.org/abs/2403.07378)**: Truncation-aware SVD with data whitening via Cholesky decomposition. Sequential layer-wise parameter update to compensate accumulated error. **LLaMA-7B at 20% compression: 1.6x inference speedup, >95% accuracy retained, 15 min to compress** (vs 5.5h for ASVD). V2 (NAACL 2025): 28% better perplexity gap, 13% higher accuracy under 7GB budget. [GitHub](https://github.com/AIoT-MLSys-Lab/SVD-LLM)
- **[ASVD (2023)](https://arxiv.org/abs/2312.05821)**: Activation-aware SVD — rescale weight matrix by activation RMS to absorb outliers before SVD. **10-30% compression without quality loss, 50% KV cache reduction**. Training-free. Slow (5.5h for 7B). [GitHub](https://github.com/hahnyuan/ASVD4LLM)
- **[AdaSVD (2025)](https://arxiv.org/abs/2502.01403)**: Adaptive per-layer rank allocation. Iterative alternating updates to compensate truncation error without training. **Consistently outperforms SVD-LLM at all compression ratios**. Key finding: early layers and specific MLP layers need higher rank. [GitHub](https://github.com/ZHITENGLI/AdaSVD)
- **[ResSVD (2025)](https://arxiv.org/abs/2505.20112)**: Two-stage SVD — truncate to rank r1, compute residual, truncate residual to rank r2. **Confirms the "truncation residual" contains recoverable structure** — directly analogous to our "base + correction" idea. Layer-selective compression (avoid early layers) is critical.

### W = Low-Rank + Sparse Decomposition

- **[LoSparse (ICML 2023)](https://arxiv.org/abs/2306.11222)**: First paper to decompose LLM weights as W = LR + S. Sparse component captures large-magnitude outlier weights that degrade low-rank approximation. **Validates the W = LR + S concept.** Limitation: requires full-network fine-tuning.
- **[HASSLE-free (CPAL 2025)](https://arxiv.org/abs/2502.00899)**: **One-shot (no fine-tuning)** post-training W = S + LR decomposition. Llama3-8B with (2:4 sparse + rank-64): **18% perplexity reduction vs prior methods, 28% accuracy gap reduction**. Joint optimization of S and LR together outperforms applying them independently.
- **[3BASiL (NeurIPS 2025)](https://arxiv.org/abs/2603.01376)**: ADMM-based algorithm with cross-layer optimization. LLaMA-8B with (2:4 sparse + rank-64): **30% better perplexity gap vs dense, 2.5x faster compression**. [GitHub](https://github.com/mazumder-lab/3BASiL)
- **[HSS Compression (CODS 2025)](https://arxiv.org/abs/2601.07839)**: Hierarchical sparse separable factorization. Uses Reverse Cuthill-McKee permutation to cluster high-magnitude weights toward diagonal, improving off-diagonal compressibility.

### Quantized + Low-Rank Combined

- **[CALDERA (NeurIPS 2024)](https://arxiv.org/abs/2405.18886)**: Decomposes W = Q + L*R where Q is quantized dense, L/R are low-rank (also quantized). **Outperforms all prior methods below 2.5 bits/parameter**. The low-rank component captures systematic error that quantization alone misses. Conceptually closest to "low-rank base + quantized corrections" — but both components always in memory. [GitHub](https://github.com/pilancilab/caldera)
- **[SLiM (ICML 2025, NVIDIA)](https://arxiv.org/abs/2410.09615)**: Three-way: quantization + 2:4 sparsity + low-rank adapter (mathematically computed, no training). **5.66% accuracy improvement over prior best, 4.3x layer-wise speedup**.

### System-Level Weight Streaming

- **[Hypura (2026, Apple Silicon)](https://github.com/t8/hypura)**: Storage-tier-aware inference scheduler placing tensors across GPU/RAM/NVMe. Uses `pread()` with `F_NOCACHE`. For dense models: attention+norms stay in GPU RAM, FFN tensors stream from NVMe via prefetch lookahead. **99.5% neuron cache hit rate after warmup.** Closest existing system to "keep compact warm weights in RAM, stream rest from SSD."
- **[ENDOR (2024)](https://arxiv.org/html/2406.11674v1)**: Hardware-friendly sparse format for SSD-offloaded inference. High compression ratio + low decompression overhead for streaming sparse weights.
- **[FlashSVD (2025)](https://arxiv.org/abs/2508.01506)**: Fuses low-rank projections directly into FlashAttention and FFN kernels. **70% reduction in peak activation memory**, zero accuracy loss. Shows fused execution of factored weights is viable.

### Singular Value Spectrum Analysis

- Weight matrices show **power-law singular value decay** on log-log plots
- **"Small singular values matter too"** [(NeurIPS 2025)](https://arxiv.org/abs/2410.17770): Random matrix theory analysis showing departures from Marchenko-Pastur distribution at both large AND small singular values, especially in Q/K/V matrices. Complicates naive truncation.
- **MLP up/down projections are more compressible than attention matrices**
- Effective rank: **32-1024 per layer** captures most useful structure (varies by compression ratio)

### Key Numbers

| Method | Decomposition | Quality | Compression | Notes |
|--------|--------------|---------|-------------|-------|
| SVD-LLM | Truncated SVD | >95% accuracy | 20-60% | Data whitening critical |
| HASSLE-free | S + LR (one-shot) | 18% better PPL gap | 2:4 sparse + rank-64 | No fine-tuning |
| 3BASiL | S + LR (ADMM) | 30% better PPL gap | 2:4 sparse + rank-64 | Cross-layer optimization |
| CALDERA | Q + LR (quantized) | SOTA below 2.5 bpw | Variable | Both terms in memory |
| SLiM | Q + sparse + LR | +5.66% accuracy | 4.3x speedup | NVIDIA hardware |
| Hypura | Tiered streaming | 99.5% cache hit | — | Apple Silicon native |

### The Gap

The end-to-end system — **decompose W = LR + S, keep LR in unified memory, stream S from SSD selectively** — has not been assembled. Components exist:
1. High-quality decomposition algorithms (HASSLE-free, 3BASiL, CALDERA)
2. System-level streaming (Hypura for Apple Silicon)
3. Execution kernels for factored weights (FlashSVD)
4. Sparse storage formats for SSD (ENDOR)

The connection is missing. For M4 with ~19GB usable: a 70B model's LR component (40-60% of original) could fit in RAM, with the S sparse corrections streamed from NVMe for layers being processed.

## Open Questions

1. ~~What rank captures "enough" of each layer?~~ **Partially answered**: ranks 32-1024 per layer, MLP more compressible than attention, early layers need higher rank (AdaSVD).
2. How do we decide when corrections are needed? ResSVD's two-stage approach suggests always applying corrections is better than selective.
3. ~~Can we combine low-rank base with quantization?~~ **Yes — CALDERA does exactly this** (W = Q + LR) and outperforms all methods below 2.5 bpw.
4. ~~What's the overhead of SVD decomposition?~~ SVD-LLM: 15 min for 7B (vs 5.5h for ASVD). 3BASiL: 2.5x faster than prior S+LR methods.
5. ~~Is the correction actually sparse enough to compress well?~~ **Yes** — LoSparse, HASSLE-free confirm sparse components are real and compressible.
6. Can Hypura's tiered storage approach be extended to stream LR vs S components rather than full layers?

## Experiment Plan

### Phase 1: Spectral analysis
- Take a 7B model, compute SVD of each weight matrix
- Plot singular value decay curves by layer type and depth
- Measure: what rank retains 90%, 95%, 99% of the Frobenius norm?

### Phase 2: Quality at reduced rank
- Run inference with rank-truncated weights (rank 64, 128, 256, 512)
- Measure: perplexity vs rank for different layer types
- Identify which layers are most sensitive to rank reduction

### Phase 3: Correction sparsity
- Compute W - BASE at various ranks
- Measure: sparsity ratio, compression ratio, distribution of correction magnitudes
- Determine: optimal storage format for corrections (CSR, block-sparse, quantized dense)

### Phase 4: End-to-end prototype
- Implement low-rank base in RAM + SSD correction loading
- Test selective correction strategies
- Measure: tok/s, perplexity, memory usage

## Risks

- If singular values decay slowly, the base won't be much smaller than the full matrix
- Correction loading from SSD may add too much latency
- SVD preparation is expensive (hours for large models)
- Accumulated errors from rank truncation across 80+ layers may compound
- Different architectures (GQA, MQA) may have different compressibility
