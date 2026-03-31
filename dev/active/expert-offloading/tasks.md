# Tasks: Expert Offloading / SSD Streaming

## Setup
- [x] Create branch (experiment/expert-offloading)
- [ ] Run baseline benchmark with largest in-memory model

## Implementation
- [ ] Phase 0: Checkpoint audit (scripts/checkpoint_audit.py)
- [ ] Phase 1: NVMe profiling script (scripts/nvme_profile.py)
- [ ] Phase 2: Page cache benchmark script (scripts/page_cache_bench.py)
- [ ] Phase 3: GPU/SSD contention test (scripts/gpu_ssd_contention.py)
- [ ] Phase 4a: Synthetic expert streaming microbenchmark (scripts/expert_stream_synthetic.py)
- [ ] Phase 4b: Real MoE integration (scripts/expert_stream.py) — stretch goal

## Validation
- [ ] Run all benchmarks
- [ ] Log results to experiments.jsonl
- [ ] Compare against baselines (dense in-memory, synthetic in-memory dequant+GEMM)

## Review
- [ ] Plan review loop completed
- [ ] Code review loop completed
- [ ] Address all high/medium findings

## Documentation
- [ ] Update context.md with findings
- [ ] Archive to dev/completed/ when done
