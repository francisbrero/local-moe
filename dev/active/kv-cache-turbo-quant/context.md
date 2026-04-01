# KV Cache Compression (TurboQuant) — Context

**Issue**: #3
**Branch**: `experiment/kv-cache-turbo-quant`
**Status**: Complete — scoped to characterization (Path A)

## Decision Gate Result

**Built-in MLX-LM KV cache quantization (`kv_bits` parameter) preserves quality perfectly and outperforms our custom TurboQuant prototype on every metric.** Scoping this issue to characterization only.

## Key Findings

### 1. MLX-LM Built-in KV Cache Quantization (Phase 0)

Tested on S (0.5B) and M (3B) tiers with FP16, 8-bit, and 4-bit KV cache:

| Tier | Config | tok/s | Avg PPL |
|------|--------|-------|---------|
| S | FP16 | 323 | 5.21 |
| S | kv8 | 245 | 5.21 |
| S | kv4 | 250 | 5.21 |
| M | FP16 | 95 | 4.13 |
| M | kv8 | 60 | 4.13 |
| M | kv4 | 38 | 4.13 |

**Zero perplexity degradation** at 4-bit KV quantization on both model sizes.

Throughput penalty at short contexts (128 tokens): 22-60% slower. This is expected — quantization overhead dominates when the cache is small. The benefit appears at longer contexts where memory savings prevent OOM.

### 2. TurboQuant vs Built-in Comparison (Synthetic Data)

Tested across S/M/XL configs, 4-bit and 8-bit, at seq_len 512/2048/8192:

| Metric | TurboQuant | Built-in | Winner |
|--------|-----------|----------|--------|
| 4-bit cosine | 0.990-0.991 | 0.991-0.993 | Built-in |
| 4-bit compression | 1.94-1.97x | 3.56x | Built-in |
| 4-bit quant latency | 8-413ms | 0.1-4.7ms | Built-in |
| 8-bit cosine | 0.9999 | 0.9999 | Tie |
| 8-bit compression | 1.94-1.97x | 1.88x | TurboQuant (marginal) |

**Built-in wins decisively at 4-bit** where KV compression matters most:
- Higher fidelity (group quantization with affine scaling adapts to local distributions)
- Better compression (native bit-packing vs our uint8 storage)
- 100-1000x faster quantization (optimized Metal kernels vs Python scalar quantization)

### 3. Why TurboQuant Rotation Doesn't Help Here

1. **MLX's group quantization is already excellent**: It uses per-group scale+bias (affine quantization), which adapts to local data distributions. The rotation's benefit (spreading information across coordinates) is less valuable when you already have fine-grained group quantization.

2. **Bit-packing gap**: Our prototype stores 4-bit indices in uint8 arrays, wasting half the bits. This makes the compression ratio look poor (1.97x vs 3.56x). Even with bit-packing, we'd reach ~3.5x, matching but not beating the built-in.

3. **Rotation overhead**: The O(d²) rotation adds significant latency (8-413ms per quantization step) with no quality benefit.

4. **Lloyd-Max global codebook vs adaptive**: The fixed Gaussian codebook works well in theory but MLX's per-group affine scaling is more practical for real data distributions.

### 4. Practical Recommendations

For the 16GB M4 target:
- **Use `kv_bits=4` with `kv_group_size=64`** — zero quality loss, ~3.5x KV cache compression
- **Focus on long contexts** where KV cache memory is the bottleneck (>4K tokens)
- **At short contexts** (<1K tokens), the quantization overhead makes it slower — skip it
- **Combine with 4-bit model weights** for maximum memory efficiency
- The built-in approach is production-ready: no custom code needed, no maintenance burden

## Implementation Summary

### Custom TurboQuant Prototype (scripts/turbo_quant/)
- `lloyd_max.py` — Lloyd-Max optimal scalar quantizer for N(0,1)
- `rotation.py` — Random orthogonal rotation via QR decomposition
- `core.py` — Quantize/dequantize with rotated-space attention
- `test_accuracy.py` — Validates roundtrip, inner product, attention fidelity
- `bench_memory.py` — Compression ratio benchmark
- `test_existing.py` — Phase 0: tests MLX-LM built-in quantization
- `bench_comparison.py` — Head-to-head comparison: TurboQuant vs built-in

### Results logged to experiments.jsonl
- Phase 0 characterization (S, M tiers)
- Long context tests (512 tokens)
- Head-to-head comparison (S, M, XL at 4-bit and 8-bit)

## Review Stats

- Plan review rounds: 7 (18+ findings addressed)
- Code review rounds: 3 (6 findings addressed: 3 high in round 1, 2 high + 1 medium in round 2, 0 new in round 3)
- Total findings addressed: 24+
