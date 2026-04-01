# KV Cache Compression (TurboQuant) — Tasks

## Phase 0: Validate Existing Implementations + Decision Gate
- [x] Implement custom TurboQuant prototype (Lloyd-Max + rotation)
- [x] Test MLX-LM built-in `QuantizedKVCache` (kv_bits parameter)
- [x] Document baseline metrics from both approaches
- [x] **Decision gate**: Built-in works perfectly — scoped to characterization (Path A)

## Phase 1: Core Algorithm Implementation (minimal baseline)
- [x] Implement Lloyd-Max scalar quantizer
- [x] Implement random orthogonal rotation via QR decomposition
- [x] Implement quantize/dequantize with rotated-space attention
- [x] Accuracy validation (roundtrip, inner product, attention fidelity)
- [ ] ~~Bit-packing~~ (deferred — built-in approach is better)
- [ ] ~~KV cache wrapper for MLX inference~~ (not needed — built-in exists)

## Phase 2: Characterization + Comparison
- [x] Built-in vs TurboQuant head-to-head comparison
- [x] Test across S/M/XL model configs
- [x] Test at various context lengths (512, 2048, 8192)
- [x] Test at 4-bit and 8-bit
- [x] Measure perplexity impact on real models (S, M tiers)
- [x] Long context generation test (512 tokens)

## Phase 3-4: Skipped
~~Memory & Performance Benchmarking~~ — Built-in already benchmarked above.
~~Architecture Interaction~~ — Built-in handles GQA natively.

## Outcome
Built-in MLX-LM KV cache quantization is production-ready with zero quality loss.
Custom TurboQuant rotation adds complexity without benefit. Recommendation: use `kv_bits=4`.
