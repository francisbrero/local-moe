# Layer LOD — Tasks

## Phase 0: Tool Validation + Environment Setup
- [ ] Install mlx-optiq
- [ ] Validate MLX mixed-quant support
- [ ] Pick and download test models (Qwen2.5-7B Q4, Q2)
- [ ] Measure baseline perplexity and memory

## Phase 1: Layer Sensitivity Profiling
- [ ] Implement `scripts/layer_sensitivity.py`
- [ ] Profile per-layer Q2 substitution on 7B model
- [ ] Profile per-layer Q3 substitution on 7B model
- [ ] Plot sensitivity curve, confirm/deny U-shape
- [ ] Log results to experiments.jsonl

## Phase 2: mlx-optiq Optimal Allocation
- [ ] Implement `scripts/optiq_allocate.py`
- [ ] Run mlx-optiq at 2.0, 2.5, 3.0, 3.5, 4.0 avg bit-widths
- [ ] Compare mlx-optiq allocations vs hand-crafted U-shape
- [ ] Quantize and measure perplexity for each allocation
- [ ] Log results to experiments.jsonl

## Phase 3: Benchmark Suite
- [ ] Implement `scripts/layer_lod_bench.py`
- [ ] Benchmark all allocations on 14B model
- [ ] Generate Pareto frontier plot
- [ ] Test across 4 task types

## Phase 4: Large Model Deployment (stretch)
- [ ] Compute 70B memory budget
- [ ] Find allocation that fits in 21GB
- [ ] Benchmark if feasible
