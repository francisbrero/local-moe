---
description: MoE expert offloading and SSD streaming patterns
globs:
  - "src/**/*.c"
  - "src/**/*.h"
  - "src/metal/**/*.metal"
alwaysApply: false
---

# MoE Expert Offloading

## Core Concept

MoE models activate only K experts per layer per token (e.g., K=4 out of 64). Store all experts on SSD, stream active ones on demand. OS page cache handles frequently used experts.

## Key Patterns

### Expert Streaming (Flash MOE)
- Parallel `pread()` system calls to load K active experts
- Expert size ~6.75MB each (4-bit quantized)
- OS page cache achieves ~71% hit rate on 48GB; lower on 16GB

### Mixed Precision Fallback (HOBBIT)
- Store both full-precision (Q4) and low-precision (Q2) copies
- Serve Q2 immediately on cache miss, Q4 when loaded
- Goal: avoid stalling on SSD reads by having a lower-quality fallback ready

### ML-Based Caching (FlashMoE paper)
- Lightweight model predicting next-needed experts
- Combines recency + frequency signals
- 51% better hit rates than LRU/LFU

### Serial Pipeline
On Apple Silicon unified memory, overlapping SSD reads with GPU compute has been reported to cause up to 73% GPU throughput degradation (observed on M2 Ultra; M4 behavior is unvalidated — see Agents.md open questions). Default to serialized pipeline, but benchmark both approaches on target hardware.

```
SSD read expert → GPU compute layer → SSD read expert → GPU compute layer
```

## Experiments Log

Track all experiments in `experiments.jsonl` with:
- Model name, expert count, active experts
- Cache hit rate, tok/s, peak memory
- Offloading strategy used

## Resources

- See issues #2 (Flash MOE), #4 (FlashMoE caching), #5 (HOBBIT)
