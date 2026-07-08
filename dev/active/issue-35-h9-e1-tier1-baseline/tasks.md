# H9-E1 Baseline — Tasks

## Phase 1: Setup
- [x] View issue #35
- [x] Read H9 plan reference
- [x] Probe environment (mlx_lm, vllm-mlx, model cache, disk)
- [x] Create branch + dev-docs

## Phase 2: Plan + review
- [x] Draft plan.md
- [ ] Plan review loop (≤10 rounds)
- [ ] User approval

## Phase 3: Implementation
- [x] Phase A: install vllm-mlx 0.3.0; serves Qwen3-30B-A3B with hermes tool parser
- [x] Phase A: confirm mlx_lm load + apply_chat_template(tools=...) works (engine gated path)
- [x] Phase A: gate_tier viable = served (vllm-mlx) and engine (mlx_lm) both available
- [x] Phase B: build scripts/h9_e1_harness.py (20 cases incl. no-tool + validators + adapters)
- [~] Phase B: run harness — BLOCKED (model load swap-limited under contention)
- [~] Phase C: decode tok/s — BLOCKED (swap-limited); cold gen did not complete to usable speed
- [~] Phase C: prefill 8K/16K — BLOCKED (same)
- [x] Phase C: memory — idle footprint 16.55 GB measured (phys_footprint); RSS undercounts to 0.37 GB
- [x] Phase D: logged under-contention record to experiments.jsonl
- [x] Phase D: gate table written to context.md (decode FAIL/contended, footprint borderline, harness blocked)
- NOTE: true decode/prefill/tool-call numbers deferred to a quiesced-machine re-run (scripts ready)

## Phase 4: Code review
- [ ] Commit WIP
- [ ] Code review loop (≤10 rounds)

## Phase 5: Finalize
- [ ] Commit with Fixes #35
- [ ] Update context.md with final results + review stats
- [ ] gh pr create
