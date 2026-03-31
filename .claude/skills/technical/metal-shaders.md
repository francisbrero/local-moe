---
description: Metal compute shader patterns for LLM inference on Apple Silicon
globs:
  - "src/metal/**/*.metal"
  - "src/**/*.m"
  - "src/**/*.swift"
alwaysApply: false
---

# Metal Shader Patterns for LLM Inference

## Key Patterns

### Fused Dequant + Matmul (Flash MOE style)
Restructure `(nibble * scale + bias) * x` as `fma(nibble, scale*x, bias*x)` to use hardware FMA units. Combines dequantization and multiplication in a single instruction — 12% performance gain.

### Unified Memory Zero-Copy
CPU-written buffers are immediately visible to GPU shaders. Use `MTLResourceStorageModeShared` for buffers that both CPU and GPU access. No explicit copy needed.

### Serial GPU/SSD Pipeline
On Apple Silicon unified memory, concurrent GPU compute and SSD DMA has been reported to reduce GPU throughput by up to 73% (observed on M2 Ultra; M4 behavior is unvalidated — see Agents.md open questions). Default to serialized: SSD read → GPU compute → SSD read → GPU compute, but benchmark both approaches on target hardware.

### Metal 4 Tensors
Metal 4 introduces tensors as first-class MSL types. Declare tensor variables directly in shaders for native tensor operations. Note: Metal 4 API is new and may have limited tooling support — verify deployment target compatibility.

## Kernel Conventions

- One kernel per .metal file
- Descriptive names: `dequant_4bit_matmul.metal`, `rmsnorm_fused.metal`
- Use threadgroup memory for shared data within a threadgroup
- Profile with Metal System Trace in Instruments

## Resources

- See issue #10 for full Metal shader research plan
- [metalQwen3](https://github.com/BoltzmannEntropy/metalQwen3) — reference implementation
- [Flash MOE](https://github.com/danveloper/flash-moe) — FMA dequant pattern
