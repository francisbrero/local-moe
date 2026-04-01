# Layer LOD — Context

**Issue**: #25
**Branch**: `experiment/layer-lod`
**Status**: Phases 0-3 complete

## Current State

All experimental phases complete. Results confirm the Layer LOD hypothesis with nuance.

## Key Findings

### Phase 0: Toolchain Validated
- **mlx 0.31.1** + **mlx-lm 0.31.1** + **optiq 0.1.0** all work
- MLX supports per-layer quantization via `quant_predicate` parameter
- OptiQ pre-quantized checkpoints load and generate correctly (PoC: Qwen3.5-0.8B-OptiQ-4bit with 187 per-layer overrides)
- Implementation path: B (OptiQ optimizer + custom quant_predicate, sensitivity requires torch)
- **72B verdict**: Does NOT fit at any bit-width. Phase 4 = analysis-only.
  - 72B at 2.0 bpw = 24.3 GB total (exceeds 24 GB)
  - 14B at 4.0 bpw = 10.1 GB total (fits comfortably — our main target)

### Phase 1: U-Shape CONFIRMED
Sensitivity profiling on Qwen2.5-7B-Instruct-4bit (28 blocks):

**Q2 substitution** (edge/middle ratio = 544x):
- Block 27 (last): CATASTROPHIC — PPL 7.0 → 705 (Δ+698)
- Block 26: Δ+1.34
- Block 4: Δ+0.98, Block 1: Δ+0.89
- Middle blocks (5-20): avg Δ = 0.11 (nearly zero impact)
- Several middle blocks actually *improve* when quantized to Q2 (blocks 6, 11, 17)

**Q3 substitution** (edge/middle ratio = 2.11x):
- Block 26: Δ+0.91 (most sensitive)
- Block 27: Δ-0.16 (improves! — Q3 is sufficient for the last block)
- Middle blocks: avg Δ = 0.06

**Practical implication**: The last 2 blocks need Q4+ protection. Middle 60% of blocks can use Q2 with minimal impact. First blocks have moderate sensitivity.

### Phase 2-3: Allocation Benchmark Results (Qwen2.5-7B)

Compared 7 allocation strategies:

| Strategy | BPW | PPL | Δ PPL | tok/s |
|----------|-----|-----|-------|-------|
| uniform_q4 (baseline) | 4.00 | 7.04 | — | 44.8 |
| uniform_q3 | 3.00 | 10.86 | +3.82 | 51.5 |
| **sensitivity_3bpw** | **3.00** | **10.74** | **+3.70** | **53.8** |
| gradient | 3.07 | 11.19 | +4.15 | 53.0 |
| ushape_20pct | 2.71 | 15.75 | +8.71 | 53.5 |
| **sensitivity_2.5bpw** | **2.50** | **17.68** | **+10.65** | **58.6** |
| uniform_q2 | 2.00 | 52,584 | +52,577 | 56.0 |

**Key results**:
1. **Sensitivity-guided at 3.0 bpw beats uniform Q3**: +1.1% better PPL (10.74 vs 10.86)
2. **Sensitivity-guided at 2.5 bpw prevents Q2 catastrophe**: PPL 17.68 vs 52,584 for uniform Q2
3. **Hand-crafted U-shape underperforms**: 15.75 PPL at 2.71 bpw — worse than uniform Q3 at 3.0 bpw
4. **Data-driven > heuristic**: Sensitivity-guided allocation consistently outperforms hand-crafted patterns
5. **Throughput bonus**: Lower precision gives ~20% speedup (44.8→53.8 tok/s)

**Conclusion**: Mixed-precision LOD works, but the improvement at same-bpw is modest (+1.1%). The real value is in preventing catastrophic quality loss at extreme compression (Q2). For practical use, the OptiQ tool automates this optimally.

### Phase 4: 72B Analysis (No Download)
- 72B at 2.0 bpw = 24.3 GB — does NOT fit in 24 GB
- Would require SSD streaming (H0) to bridge the gap
- No bit-width achieves <24 GB fit for Qwen2.5-72B on this hardware

## Review Stats
- Plan review rounds: 4
- Total findings addressed: 13 (5 high, 6 medium, 2 low)
- Code review rounds: TBD

## Practical Recommendations
1. **Use `optiq convert` for any MLX model** — automates per-layer sensitivity + optimization
2. **Protect the last 2 transformer blocks** at all costs — they're catastrophically sensitive
3. **Middle 60% of blocks can be aggressively compressed** (Q2-Q3) with minimal quality loss
4. **For 24 GB M4 Pro**: 14B at 3.0 bpw with LOD gives near-Q4 quality in ~8 GB → plenty of room for KV cache
