---
description: Weight and KV cache quantization techniques for extreme compression
globs:
  - "src/**/*.c"
  - "src/**/*.h"
  - "scripts/**/*quant*"
alwaysApply: false
---

# Quantization Techniques

## Weight Quantization

### 4-bit (GGUF Q4_K_M)
Standard for consumer hardware. 70B model ≈ 40GB. Too large for 16GB.

### 2-bit (QuIP#, AQLM)
Post-training quantization. 70B model ≈ 17GB. Tight fit on 16GB with KV cache.
- QuIP#: Hadamard incoherence + lattice codebooks
- AQLM: Additive quantization via learned codebooks

### 1.58-bit (BitNet)
Ternary weights {-1, 0, +1}. 70B model ≈ 14GB. Requires natively trained models.
- bitnet.cpp for inference
- No multiply operations — just additions and subtractions

## KV Cache Quantization (TurboQuant)

Compress KV cache to 3-4 bits with zero accuracy loss:
1. Random rotation of vectors
2. Per-dimension quantization
3. 1-bit QJL residual correction

6x memory reduction. Training-free, data-oblivious.

## Memory Budget (16GB M4)

| Component | Budget |
|-----------|--------|
| OS + overhead | ~5GB |
| Non-expert weights | 2-4GB |
| Metal scratch buffers | ~200MB |
| KV cache | 1-3GB (before compression) |
| Expert cache | remainder |

## Resources

- See issue #7 (extreme quantization) and issue #3 (TurboQuant)
