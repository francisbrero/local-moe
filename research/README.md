# Research: Novel Approaches for Large LLM Inference on Constrained Hardware

## The Problem

Running large LLMs (70B+) on a 24GB M4 MacBook Pro. The model weights alone exceed available memory, and memory bandwidth caps throughput even when weights fit.

## Three Bottlenecks


| Bottleneck           | Constraint                    | Hard Number                      |
| -------------------- | ----------------------------- | -------------------------------- |
| **Memory capacity**  | 24GB unified, ~19-21GB usable | Q4 70B = 40GB weights            |
| **Memory bandwidth** | ~120 GB/s on M4               | Max ~3 tok/s for 40GB model      |
| **Compute**          | ~4 TFLOPS FP32                | Rarely the bottleneck at batch=1 |


## Hypotheses

Each hypothesis draws an analogy from another field of computer science where similar constraints were overcome.


| #   | Hypothesis                                                            | Analogy                        | Prior Art                                                                                 | Actionability                                                  |
| --- | --------------------------------------------------------------------- | ------------------------------ | ----------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| H0  | [Expert offloading / SSD streaming](./h0-expert-offloading.md)        | Virtual memory / demand paging | Flash MOE, HOBBIT (9.93x), FlashMoE (51% better cache)                                    | **Tested — Promising. 30B on 16GB confirmed.**                 |
| H1  | [Adaptive precision per token](./h1-adaptive-precision.md)            | Adaptive bitrate streaming     | FlexQuant (1.3x), EAD (41-67% compute reduction), ML-SpecQD (2.72x), JANG (Apple Silicon) | Medium — needs custom implementation                           |
| H2  | [Expert routing prediction + prefetch](./h2-routing-prediction.md)    | CPU branch prediction          | 12+ papers, 93-97% prediction accuracy, up to 5x speedup (ProMoE)                         | High — well-validated, clear implementation path               |
| H3  | [Prompt-aware weight pre-loading](./h3-prompt-aware-loading.md)       | Database query planning        | SiDA-MoE (3.93x), eMoE (80% mem reduction), expert specialization confirmed               | Medium — building blocks exist, end-to-end gap                 |
| H4  | [Low-rank base + sparse SSD corrections](./h4-lowrank-corrections.md) | Compressed sensing             | SVD-LLM (1.6x), HASSLE-free, CALDERA (<2.5bpw SOTA), Hypura (Apple Silicon)               | Medium — decomposition solved, system gap                      |
| H5  | [Layer LOD — variable precision by depth](./h5-layer-lod.md)          | Game engine LOD                | U-shape confirmed, mlx-optiq (2.3x accuracy), llama.cpp regex quant, CoopQ                | **Highest — immediately actionable with existing tools**       |
| H6  | [Activation sparsity prediction](./h6-activation-sparsity.md)         | Frustum culling                | PowerInfer (27.8x), TEAL (1.5-1.8x, training-free), TurboSparse (90% sparsity)            | **High — TEAL requires zero training, CMoE converts in 5 min** |
| H7  | [KV cache compression](./h7-kv-cache-compression.md)                  | Video codec lossy compression  | TurboQuant (Google, ICLR 2026), MLX-LM built-in QuantizedKVCache                           | **Complete — built-in `kv_bits=4` works perfectly, zero quality loss** |


## How to Use This

1. Read a hypothesis page to understand the idea, prior art, and open questions
2. Design an experiment (see `dev/templates/plan.md`)
3. Run it using the experiment harness (`scripts/run.py`)
4. Log results to `experiments.jsonl`
5. Update the hypothesis page with findings

## Status Key

- **Untested** — Idea documented, no experiments run
- **In Progress** — Active experimentation
- **Promising** — Positive results, worth pursuing further
- **Mixed** — Some benefit, significant trade-offs
- **Dead End** — Tested, doesn't work for our constraints

