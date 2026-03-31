# H5: Layer LOD — Variable Precision by Depth

**Status**: Untested
**Analogy**: Game engine Level of Detail (LOD)
**Bottleneck addressed**: Memory capacity + bandwidth

## The Insight

In 3D game engines, objects close to the camera are rendered with high-polygon meshes (high detail) while distant objects use simplified meshes (low detail). The player never notices because visual importance decreases with distance. This saves massive amounts of GPU computation and memory.

Transformer layers have a similar property: **not all layers contribute equally to output quality**. Research on layer pruning shows:
- Early layers (embedding, first few transformers) build fundamental representations — high impact
- Late layers (final few before output head) shape the distribution over tokens — high impact
- Middle layers are often redundant — experiments show you can skip or compress many of them with minimal quality loss

## Hypothesis

> By assigning higher precision (Q8/Q4) to the first and last N layers and aggressive compression (Q2/Q1) to middle layers, we can fit significantly larger models in RAM while maintaining output quality.

## Mechanism

```
Layer allocation for a 70B model (80 layers):

  Layers 0-9   (first 10):  Q4  — 5GB  (high detail, close to input)
  Layers 10-69 (middle 60): Q2  — 12GB (low detail, "far away")
  Layers 70-79 (last 10):   Q4  — 5GB  (high detail, close to output)
  Embeddings + output head:  Q6  — 2GB  (critical)

  Total: ~24GB — fits in RAM!
```

Compare to uniform Q4 (40GB) or uniform Q2 (17GB + quality loss).

## Why This Might Work

Multiple lines of evidence suggest middle layers are over-parameterized:

1. **Layer pruning studies**: Removing 30-50% of middle layers from LLMs degrades perplexity by only 1-3 points
2. **Layer distillation**: Middle layers can be distilled into fewer layers with minimal loss
3. **Residual connections**: The residual stream carries information past middle layers, reducing their marginal contribution
4. **Cosine similarity**: Hidden states between adjacent middle layers have very high cosine similarity (~0.99), suggesting incremental changes

## Expected Impact

- **Memory**: 70B model fits in ~24GB with mixed Q4/Q2 allocation
- **Quality**: Better than uniform Q2, close to uniform Q4 (because critical layers keep precision)
- **Bandwidth**: Mixed precision reduces total bytes read per token
- **No SSD needed**: If the model fits in RAM, we avoid SSD latency entirely

## Prior Art (Research Findings)

**The U-shaped importance pattern is confirmed** across multiple independent studies. This is one of the most well-validated hypotheses. Existing tools already support it.

### Direct Confirmation of U-Shape

- **[Layer-Sensitive Quantization (2025)](https://arxiv.org/html/2503.06518v1)**: Measured per-layer quantization sensitivity across Llama, Mistral, Qwen, Gemma. **"Sensitivity spikes tend to be present at the start and end layers."** Pattern holds across datasets, quant methods, bit-widths, and fine-tuned variants. SensiBoost/KurtBoost methods give extra bits to boundary layers: **up to 9% perplexity reduction at 3-bit with only 2% additional memory**.
- **[Variable Layer-Wise Quantization (2024)](https://arxiv.org/html/2406.17415v1)**: Used LIM (Layer Input Modification) and ZD (Z-score Distribution) metrics. **"The first and the last layer are the most two important layers. Lesser important layers tend to be towards halfway of middle and end."** Llama-2-13B: 30 layers at 4-bit, 10 layers at 2-bit (3.5 avg bpw). Performance stable until 25-50% of layers at lower precision (vs 5-10% with random ordering).

### Layer Pruning Evidence

- **[ShortGPT (ACL 2025)](https://arxiv.org/html/2403.03853v3)**: Block Influence (BI) metric — cosine similarity between layer input/output. **Most-removed layers concentrated in upper-middle** (Llama-2-7B: layers 27, 26, 25, 28...; Llama-2-13B: layers 33, 31, 32, 30...). "Redundancy is primarily manifested in the middle to later layers." **Removing 25% of parameters retains ~91.6% performance** on Llama-2-13B. Up to 55% of layers prunable on multiple-choice tasks.
- **[Reassessing Layer Pruning (2024)](https://arxiv.org/html/2411.15558v1)**: Simply removing the final 25% of layers outperformed BI and Taylor methods (**52.68% vs 40.80% accuracy**). But lm_head must be tuned after pruning — the output projection is separately critical.

### Two-Phase Model of Layer Function

- **[Attend First, Consolidate Later (BlackboxNLP 2024)](https://arxiv.org/html/2409.03621v1)**: Manipulation experiments on 4 LLMs. **Early layers (~first 50-70%): information-gathering phase** — corrupting hidden states causes catastrophic loss. **Late layers (~top 30-50%): internal-processing phase** — replacing with random vectors causes "small to negligible drop." Pattern: early layers critical, large middle-to-late stretch is robust, final output projection separately important.

### Task-Dependent Layer Roles (Caveat)

- **[Demystifying Layer Roles (2025)](https://arxiv.org/abs/2510.02091)**: **Retrieval/knowledge: concentrated in shallow layers** (ablating early layers causes -0.8 accuracy). **Reasoning: distributed across mid-to-deep layers** (GSM8K sensitive to layers 6, 23, 35 on Qwen3-8B). **Warning: the importance pattern is task-dependent** — not a uniform U-shape for all capabilities. Likelihood-based evaluations (multiple choice) make middle layers appear more dispensable than they are for generation tasks.

### Existing Tools That Support Per-Layer Quantization

| Tool | Per-Layer Support | Platform | Notes |
|------|------------------|----------|-------|
| **llama.cpp / GGUF** | Regex-based per-tensor quantization | All | Q2_K already uses Q4_K for attention.vw + feed_forward.w2. Can be manually specified. |
| **EXL2 / ExLlamaV2** | Automatic per-matrix bit allocation | NVIDIA | Measures quant error per-matrix against calibration data. Closest practical implementation to LOD. |
| **[mlx-optiq](https://mlx-optiq.pages.dev/)** | KL-divergence sensitivity per layer | **Apple Silicon** | Greedy knapsack solver assigns optimal per-layer bit-widths. **Qwen3.5-0.8B: 2.3x better accuracy than uniform 4-bit, -2% speed penalty**. |
| **MLX native** | Mixed 3/4/6/8-bit per layer | Apple Silicon | Common practice: 6-bit embeddings, 4-bit body. |
| **[AutoRound (Intel, 2025)](https://github.com/intel/auto-round)** | Mixed-precision in minutes | All | Exports to GGUF, GPTQ, AWQ. |
| **[CoopQ (2025)](https://arxiv.org/html/2509.15455)** | Shapley values for inter-layer interactions | All | **20-80% perplexity reduction** vs best baseline at sub-4-bit. Accounts for layer interaction effects. |
| **[LSAQ (2024)](https://arxiv.org/html/2412.18135v1)** | Jaccard similarity metric | All | Cheapest metric (no gradients). High Jaccard = layer barely changes tokens = lower precision OK. Outperformed uniform in 90% of cases. |

### Key Numbers

| Finding | Evidence | Source |
|---------|----------|--------|
| U-shape confirmed | "Sensitivity spikes at start/end layers" | arxiv 2503.06518 |
| 9% PPL improvement | Protecting boundary layers at 3-bit, 2% extra memory | arxiv 2503.06518 |
| 25% layers removable | 91.6% performance retained | ShortGPT |
| 55% layers removable | On multiple-choice tasks | ShortGPT |
| Middle-to-late layers most redundant | BI metric, cosine similarity ~0.99 | ShortGPT |
| mlx-optiq 2.3x accuracy improvement | vs uniform 4-bit, on Apple Silicon | mlx-optiq |
| CoopQ 20-80% PPL reduction | vs best baseline at sub-4-bit | arxiv 2509.15455 |

### Practical Recommendation for M4

**Immediately actionable** — three paths require no custom code:
1. **llama.cpp regex quant**: Manually assign Q4_K to first/last N layers, Q2_K to middle. Tooling exists.
2. **mlx-optiq**: Automatic calibration-driven per-layer allocation on Apple Silicon. Compare its bit assignments to the U-shape hypothesis.
3. **LSAQ Jaccard metric**: Cheapest to implement (no gradients), apply to any GGUF model.

## Open Questions

1. ~~What's the optimal precision allocation? Is it really U-shaped?~~ **Confirmed U-shaped** across multiple models (arxiv 2503.06518, 2406.17415).
2. ~~Does the optimal allocation depend on the task?~~ **Yes** — retrieval/knowledge needs early layers, reasoning needs mid-to-late layers (arxiv 2510.02091). A single U-shaped allocation may not be optimal for all tasks.
3. ~~Can we determine the allocation automatically?~~ **Yes** — mlx-optiq (KL-divergence), CoopQ (Shapley values), LSAQ (Jaccard similarity) all do this.
4. How does this interact with GQA? Attention heads may have different sensitivity than FFN.
5. What about MoE models — should expert layers get different LOD than attention layers? (JANG from H1 suggests attention at 8-bit, experts at 2-3 bit.)
6. Can we combine LOD quantization with layer streaming (H5 + Approach 8 from Agents.md) — keep first/last layers in RAM at Q4, stream middle layers from SSD at Q2?

## Experiment Plan

### Phase 1: Layer sensitivity profiling
- Take a 7B model, quantize one layer at a time to Q2 while keeping others at Q4
- Measure per-layer perplexity impact
- Plot sensitivity curve by layer depth

### Phase 2: Block sensitivity
- Group layers into blocks of 5-10
- Quantize each block to Q2 while keeping others at Q4
- Measure: which blocks can tolerate Q2 without significant quality loss?

### Phase 3: Optimal allocation search
- Test several allocation patterns:
  - Uniform Q4 (baseline)
  - Uniform Q2 (lower bound)
  - U-shaped (Q4 edges, Q2 middle)
  - Gradient (Q4 → Q3 → Q2 → Q3 → Q4)
  - Sensitivity-guided (per-layer based on Phase 1)
- Measure: perplexity vs total memory for each allocation

### Phase 4: Large model deployment
- Apply best allocation to 70B model
- Measure: does it fit in 24GB? Perplexity vs uniform quantization?
- Test across multiple tasks (code, math, creative, chat)

## Risks

- The U-shaped hypothesis may be wrong — sensitivity could be uniform or erratic
- Perplexity may not capture real quality differences (some layers may be critical for specific capabilities like math)
- Mixed-precision dequantization is more complex — may need separate Metal kernels per precision level
- GGUF format may not support per-layer quantization (need custom format or MLX)
