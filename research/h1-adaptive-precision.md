# H1: Adaptive Precision Per Token

**Status**: Untested
**Analogy**: Adaptive bitrate streaming (Netflix, YouTube)
**Bottleneck addressed**: Memory bandwidth

## The Insight

In adaptive bitrate video streaming, the server sends low-resolution frames when bandwidth is tight and high-resolution when it's available. The viewer rarely notices because most frames are predictable.

LLM token generation has the same property: **most tokens are "easy"** (high confidence, low entropy). The model could predict "the" or "of" with Q2 weights just as well as Q8. Only ambiguous tokens — where the top-k probabilities are close — actually benefit from higher precision.

## Hypothesis

> If we use Q2 weights for high-confidence tokens and dynamically load Q4/Q8 weights only for low-confidence tokens, we can achieve near-Q4 quality at near-Q2 bandwidth cost.

## Mechanism

```
For each token:
  1. Run forward pass with Q2 weights (fast, in RAM)
  2. Check output entropy / top-k confidence gap
  3. If confident (entropy < threshold): accept token, move on
  4. If uncertain (entropy ≥ threshold):
     a. Load Q4/Q8 weights for critical layers from SSD
     b. Re-run those layers at higher precision
     c. Accept the refined token
```

## Expected Impact

- **Bandwidth savings**: If 70-80% of tokens are "easy", we only need high-precision weights for 20-30% of tokens
- **Effective throughput**: 3-5x over running everything at Q4
- **Quality**: Near-Q4 perplexity since hard tokens still get full precision

## Prior Art (Research Findings)

### Token Entropy Distribution — The Empirical Foundation

**~80% of tokens are "easy"** — this is now well-established:
- [Beyond the 80/20 Rule (Wang et al., 2025)](https://arxiv.org/abs/2506.01939): >50% of CoT tokens have near-zero entropy (H_t < 0.01). Only the top ~20% are genuine "forking tokens" (H_t > 0.672). Training on only the low-entropy 80% produced "marked performance decline" — confirming they're predictable.
- [DiffAdapt (Liu et al., 2025)](https://arxiv.org/abs/2510.19669): Up to 62% token reduction via difficulty-adaptive reasoning. Uses a lightweight hidden-state probe to classify easy/normal/hard tokens.

### Dynamic Per-Token Precision Switching

- **[FlexQuant (2025)](https://arxiv.org/abs/2506.12024)**: The closest paper to our exact idea. Monitors entropy + KL divergence per token step, switches between INT8 and INT4 layers. **1.3x speedup** over static INT8, ROUGE-L within ~2-3% of full precision. Limitation: only 4-bit/8-bit, modest gains, switching overhead limits benefit.
- **[FlexQuant — edge variant (2025)](https://arxiv.org/abs/2501.07139)**: Addresses memory fluctuation on edge devices. Generates an ensemble of quantized model variants, swaps between them based on runtime memory pressure. 15x granularity improvement. Memory-pressure-driven, not token-difficulty-driven, but similar mechanism.
- **[DTQAtten (DATE 2022)](https://ieeexplore.ieee.org/document/9774692/)**: Two-level top-K engine for attention: full precision for top tokens, 4-bit for middle tier, 0-bit (discarded) for bottom. 40.5% average 4-bit ratio. Encoder-only, but insight transfers.

### Entropy-Based Model/Precision Switching

- **[Entropy Adaptive Decoding (EAD, 2025)](https://arxiv.org/abs/2502.06833)**: Routes between small and large model based on rolling entropy. **96.7% of 11B performance while using the large model for only 43% of tokens (41.5% compute reduction)**. On Qwen: large model used for only 25% of tokens (67% compute reduction). Does NOT verify — accepts controlled quality divergence.
- **[ML-SpecQD (2025)](https://arxiv.org/abs/2503.13565)**: Uses MXFP4 quantized version of same model as draft in speculative decoding. **71.2% acceptance rate** (vs 41.7% for a separate small draft). **2.72x speedup** with three-level hierarchy. This IS adaptive precision expressed as speculative decoding.
- **[Entropy-Aware Speculative Decoding (2024)](https://arxiv.org/abs/2512.23765)**: When both draft and target have high entropy, reject the draft. Actually surpasses target model quality in some cases by catching uncertainty cascades.

### Early Exit / Adaptive Depth (Adjacent Approach)

- **[CALM (Google, NeurIPS 2022)](https://arxiv.org/abs/2207.07061)**: Tokens exit at intermediate layers when confidence exceeds threshold. **Up to 3x speedup** on summarization. Challenge: KV cache consistency for tokens that exit early.
- **[SpecEE (ISCA 2025)](https://arxiv.org/abs/2504.08850)**: Speculative early exiting with lightweight predictor. **2.25x speedup on Llama2-7B** (cloud), **2.43x (edge/llama.cpp)**. Has llama.cpp implementation — directly applicable to M4.
- **[ADEPT (2026)](https://arxiv.org/abs/2601.03700)**: Extends early exit to both prefill and generation. Addresses KV cache problem for skipped layers.

### Per-Component Static Mixed Precision (Apple Silicon)

- **[JANG/jangq (2025)](https://github.com/jjang-ai/jangq)**: MLX-native, Apple Silicon optimized. Attention layers at 8-bit (1-5% of params), expert/MLP at 2-3 bit (95%+ of params). **Qwen3.5-397B at 2.1-bit avg: 86.5% MMLU**. MLX produces NaN below 4-bit on the same model without this technique. Proves the precision differentiation principle works on Apple Silicon.
- **[MoQAE (ACL 2025)](https://arxiv.org/abs/2506.07533)**: Treats different quantization bit-widths as "experts", routes token chunks to different quant configs via a trained router.

### Key Numbers

| Metric | Value | Source |
|--------|-------|--------|
| Easy tokens in CoT | ~80% | arxiv 2506.01939 |
| Large model needed (LLaMA) | 43% of tokens | EAD |
| Large model needed (Qwen) | 25% of tokens | EAD |
| CALM speedup | up to 3x | arxiv 2207.07061 |
| SpecEE speedup (llama.cpp) | 2.43x | arxiv 2504.08850 |
| FlexQuant speedup | 1.3x | arxiv 2506.12024 |
| ML-SpecQD speedup (3-level) | 2.72x | arxiv 2503.13565 |
| MXFP4 draft acceptance rate | 71.2% | arxiv 2503.13565 |

### The Gap

The exact mechanism — token-by-token dynamic weight dequantization at Q2 vs Q4 vs Q8 based on real-time entropy — has **not** been cleanly implemented end-to-end. FlexQuant is closest but limited to 4/8-bit with modest gains. ML-SpecQD achieves the effect via speculative decoding but is hardware-specific (MXFP4). JANG proves per-component precision works on Apple Silicon but is static.

The missing piece: a system that computes token entropy during generation, gates between Q2/Q4 weights for the same model per-token, and maintains KV cache consistency. The 80/20 distribution data suggests 40-67% compute reduction is achievable.

## Open Questions

1. What's the right confidence threshold? Too low = quality loss. Too high = no savings.
2. How fast can we load higher-precision weights from SSD for a subset of layers? If the overhead of switching precision exceeds the savings, this doesn't work.
3. Do we need both Q2 and Q4 copies of all weights, or can we store a Q2 base + Q4 "correction" deltas?
4. Which layers benefit most from higher precision on hard tokens? Probably attention > FFN (confirmed by JANG).
5. ~~What fraction of tokens in typical workloads (chat, code, reasoning) are "easy"?~~ **Answered: ~80% are easy** (arxiv 2506.01939).
6. Can ML-SpecQD's approach (quantized draft of same model) be adapted for Apple Silicon / Metal?
7. Is SpecEE (early exit, has llama.cpp implementation) a more practical starting point than full precision switching?

## Experiment Plan

### Phase 1: Token difficulty profiling
- Run a reference model at FP16 and record per-token entropy
- Classify tokens as easy/hard at various thresholds
- Measure: what % of tokens are easy for different text types?

### Phase 2: Precision sensitivity
- Compare Q2 vs Q4 vs Q8 output on easy vs hard tokens
- Measure: does Q2 match Q4 on easy tokens? How much does Q4 help on hard tokens?

### Phase 3: Dynamic switching prototype
- Implement adaptive precision in MLX or custom Metal
- Measure: tok/s, perplexity, switching overhead

## Risks

- SSD read latency for loading higher-precision weights may negate speed gains
- Token-level granularity may be too fine — batch-level or window-level might be more practical
- Need to store 2 copies of weights (Q2 + Q4), increasing storage requirements
