# Research Approaches: Running Large LLMs on 16GB M4 MacBook Pro

> **AI agents**: For project instructions and conventions, see [CLAUDE.md](./CLAUDE.md).

## Goal

Explore and test techniques to run the largest possible LLM inference on a 16GB M4 MacBook Pro, drawing from Flash MOE, TurboQuant, and related research. We are willing to rewrite anything and go as low-level as needed (custom Metal shaders, C, etc.) to make this work.

## Hardware Constraints

- **Device**: MacBook Pro M4, 16GB unified memory
- **GPU/CPU**: Shared unified memory (no discrete GPU, no data copy overhead)
- **Storage**: NVMe SSD (~5-7 GB/s read bandwidth)
- **Key advantage**: Unified memory architecture means zero-copy CPU/GPU buffer sharing
- **Key limitation**: 16GB must be shared between OS, active model weights, KV cache, and Metal scratch buffers

---

## Approach 1: Expert Offloading / Streaming (Flash MOE Style)

### Source: [Flash MOE](https://github.com/danveloper/flash-moe)

**What it does**: Runs Qwen3.5-397B (209GB) on a 48GB MacBook by streaming MoE expert weights from SSD on demand through custom Metal compute shaders. Only K=4 active experts (~6.75MB each) are loaded per layer. The OS page cache handles caching with ~71% hit rates.

**Key techniques**:
- Expert streaming from NVMe with parallel system calls
- FMA-optimized 4-bit dequantization in Metal shaders (12% gain)
- Hand-written Metal kernels for matmul, SwiGLU, RMSNorm, RoPE, MoE routing
- Serial GPU compute / SSD I/O (concurrent access degrades GPU throughput by 73% on unified memory)
- "Trust the OS" page cache strategy beat custom LRU caches and prefetching
- 58 experiments showed many intuitive optimizations (LZ4 compression, predictive prefetch) actually hurt performance

**Adaptation for 16GB**: With only ~10GB usable after OS overhead, we'd need more aggressive quantization (2-3 bit) on expert weights, and careful memory budgeting. The core streaming architecture remains valid — the SSD bandwidth is the same regardless of total RAM.

**Priority**: HIGH — This is our primary reference architecture.

---

## Approach 2: KV Cache Compression (TurboQuant)

### Source: [Google Research — TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)

**What it does**: Compresses the KV cache to 3-4 bits per element with zero accuracy loss. 6x memory reduction and up to 8x inference speedup. Training-free and data-oblivious — works on any existing model.

**Key techniques**:
- Random rotation of data vectors to simplify geometry
- Standard quantizer applied per-dimension after rotation
- 1-bit QJL residual pass to remove inner-product estimation bias
- No retraining or fine-tuning required

**Why it matters for us**: On 16GB, KV cache can dominate memory at longer contexts. A 6x reduction means we can either run longer contexts or free memory for larger model weights. This is complementary to expert offloading.

**Priority**: HIGH — Direct memory savings, composable with other approaches.

**References**:
- [MarkTechPost overview](https://www.marktechpost.com/2026/03/25/google-introduces-turboquant-a-new-compression-algorithm-that-reduces-llm-key-value-cache-memory-by-6x-and-delivers-up-to-8x-speedup-all-with-zero-accuracy-loss/)
- [VentureBeat coverage](https://venturebeat.com/infrastructure/googles-new-turboquant-algorithm-speeds-up-ai-memory-8x-cutting-costs-by-50)
- Presented at ICLR 2026 and AISTATS 2026

---

## Approach 3: ML-Based Expert Caching (FlashMoE — the paper)

### Source: [FlashMoE (arXiv:2601.17063)](https://arxiv.org/abs/2601.17063)

**What it does**: SSD-based expert offloading with an ML-driven cache replacement strategy that adaptively combines recency and frequency signals. Up to 51% better cache hit rates than LRU/LFU, 2.6x speedup over existing MoE inference systems.

**Key difference from Flash MOE repo**: This is a research paper focused on smarter caching policies rather than hand-tuned Metal shaders. The two are complementary — use FlashMoE's ML cache logic with Flash MOE's Metal pipeline.

**Priority**: MEDIUM — Improves cache hit rates which directly reduces SSD reads.

---

## Approach 4: Mixed Precision Expert Offloading (HOBBIT)

### Source: [HOBBIT (arXiv:2411.01433)](https://arxiv.org/abs/2411.01433)

**What it does**: Dynamically replaces cache-miss experts with low-precision versions to reduce loading latency while preserving accuracy. Up to 9.93x decoding speedup over existing MoE offloading systems.

**Key techniques**:
- Token-level dynamic expert loading
- Layer-level adaptive expert prefetching
- Sequence-level multidimensional expert caching
- Built on top of llama.cpp

**Why it matters**: Instead of waiting for a full-precision expert to load from SSD, serve a low-precision version immediately and upgrade later. Particularly valuable on 16GB where cache misses are more frequent.

**Priority**: HIGH — Directly addresses our memory constraint with a smart fallback strategy.

---

## Approach 5: CPU/GPU Hybrid Inference (KTransformers)

### Source: [KTransformers (GitHub)](https://github.com/kvcache-ai/ktransformers)

**What it does**: Heterogeneous CPU/GPU inference for MoE models. Routes expert FFN computation to CPU while attention runs on GPU. 4.62-19.74x prefill speedup, 1.25-4.09x decode speedup.

**Key techniques**:
- AMX-specialized CPU kernels (Intel-focused, would need Metal/Accelerate adaptation)
- Asynchronous CPU-GPU task scheduling
- Expert weights in DRAM, attention on GPU

**Adaptation needed**: KTransformers targets Intel AMX + NVIDIA GPU. We'd need to rewrite CPU kernels for Apple's AMX/Accelerate and GPU kernels for Metal. The scheduling architecture is the valuable part.

**Priority**: MEDIUM — Good architecture but requires significant porting work.

---

## Approach 6: Extreme Quantization (2-bit / 1-bit)

### Sources:
- [BitNet (Microsoft)](https://github.com/microsoft/BitNet) — 1.58-bit ternary weights {-1, 0, +1}
- [QuIP# (arXiv)](https://arxiv.org/abs/2307.13304) — 2-bit post-training quantization with Hadamard incoherence
- AQLM — Additive quantization via learned codebooks, Pareto-optimal sub-3-bit

**What it does**: Compress model weights to 1-2 bits, reducing a 70B model from ~140GB (FP16) to ~9-17GB. BitNet runs 100B models on CPU alone at 5-7 tok/s.

**Key insight**: At 2-bit quantization, a 70B parameter model fits in ~17GB. At 1.58-bit (BitNet), it's even smaller. This could let us fit models that are otherwise impossible on 16GB.

**Trade-offs**: Post-training quantization (QuIP#, AQLM) works on existing models but loses some accuracy at extreme compression. Native 1-bit training (BitNet) preserves accuracy but requires specially trained models.

**Priority**: HIGH — Potentially the most impactful single technique for our memory budget.

---

## Approach 7: Speculative Decoding

### Sources:
- [Apple Mirror-SD](https://machinelearning.apple.com/research/mirror) — 2.8-5.8x wall-time speedup
- [Apple ReDrafter](https://machinelearning.apple.com/research/recurrent-drafter) — Up to 2.3x speedup, implemented in MLX

**What it does**: Use a small "draft" model to generate candidate tokens, then verify them in parallel with the large model. Multiple tokens can be accepted per forward pass.

**Why it matters**: Doesn't reduce memory — but dramatically improves tokens/second. If we can fit the model in memory (via quantization + offloading), speculative decoding multiplies our throughput.

**Priority**: MEDIUM — Throughput multiplier, but only after we solve the memory problem.

---

## Approach 8: Layer-wise Streaming Inference

### Sources:
- [Apple — LLM in a Flash (arXiv:2312.11514)](https://arxiv.org/abs/2312.11514)
- [Layer-wise inferencing + batching](https://verdagon.dev/blog/llm-throughput-not-ram-limited) — 7x speedup on 16GB Mac

**What it does**: Stream transformer layers one at a time from SSD to GPU, process all tokens for that layer, then evict it. Only one layer's weights need to be in memory at a time.

**Key insight**: This is the dense model equivalent of MoE expert offloading. Instead of loading all layers, stream them. With batching, a 16GB Mac achieved 7x speedup.

**Why it matters**: For dense models (non-MoE), this is the primary technique to exceed memory. Works with any architecture.

**Priority**: MEDIUM — Fallback for dense models; MoE streaming is more efficient for MoE architectures.

---

## Approach 9: Custom Metal Shaders + Metal 4

### Sources:
- [Metal 4 — WWDC 2025](https://developer.apple.com/metal/)
- [metalQwen3](https://github.com/BoltzmannEntropy/metalQwen3) — Full Qwen3 in Metal compute shaders
- [Flash MOE](https://github.com/danveloper/flash-moe) — Hand-written Metal kernels

**What it does**: Write inference kernels directly in Metal Shading Language instead of using frameworks. Metal 4 introduces tensors as first-class shader citizens and allows AI inference directly in shaders.

**Key techniques**:
- Fused dequantization + matmul kernels (avoid separate passes)
- FlashAttention in Metal
- Custom memory management via MTLHeap
- Exploiting unified memory zero-copy

**Priority**: HIGH — This is the execution layer that makes everything else fast.

---

## Approach 10: MLX Framework Optimizations

### Sources:
- [MLX (Apple)](https://github.com/ml-explore/mlx)
- [WWDC 2025 sessions on MLX](https://developer.apple.com/videos/play/wwdc2025/298/)
- [vllm-mlx](https://arxiv.org/html/2601.19139v2)

**What it does**: Apple's ML framework optimized for Apple Silicon. Mixed-precision quantization (4-bit body, 6-bit embeddings), kernel fusion via `mx.compile`, `mx.fast` for optimized RMSNorm/attention.

**Key advantage**: First-party Apple support, optimized for unified memory, growing ecosystem.

**Trade-off vs custom Metal**: MLX is higher-level and easier to iterate with, but custom Metal shaders can be faster for specific operations. Good for prototyping before going low-level.

**Priority**: MEDIUM — Good prototyping platform, may be sufficient without going full Metal.

---

## Recommended Experiment Plan

### Phase 1: Baseline & Quick Wins
1. Run a large MoE model (Qwen3 MoE, DeepSeek-V3) with llama.cpp + MoE offloading on 16GB
2. Apply TurboQuant-style KV cache compression
3. Test extreme quantization (2-bit GGUF) of a 70B dense model
4. Benchmark MLX vs llama.cpp for our hardware

### Phase 2: Custom Streaming Pipeline
5. Port Flash MOE's Metal streaming architecture, adapted for 16GB memory budget
6. Implement HOBBIT-style mixed precision fallback for cache misses
7. Add FlashMoE's ML-based cache replacement policy
8. Write fused dequant+matmul Metal shaders for 2-3 bit weights

### Phase 3: Advanced Optimizations
9. Implement speculative decoding on top of the streaming pipeline
10. Explore layer-wise streaming for dense models
11. Test BitNet-style 1-bit models if available in target architectures
12. Profile and optimize the serial GPU/SSD pipeline for M4 specifically

### Phase 4: Integration
13. Combine best techniques into a unified inference engine
14. Build benchmarking harness comparing all approaches
15. Document what works and what doesn't (Flash MOE's 58-experiment approach)

---

## Key Open Questions

- What is the actual usable memory on a 16GB M4 after OS and Metal overhead? (Likely ~10-11GB)
- Can we combine TurboQuant KV compression with 2-bit weight quantization without compounding accuracy loss?
- Does the M4's NVMe controller behave differently than the M2/M3 for streaming workloads?
- Is Metal 4's tensor support mature enough for production inference kernels?
- What's the largest model we can realistically run at interactive speeds (>1 tok/s)?
