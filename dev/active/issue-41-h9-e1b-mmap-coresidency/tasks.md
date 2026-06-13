# H9-E1b Tasks (#41)

## Phase 1 — Setup
- [x] View issue #41, restore E1 context
- [x] Commit #35 dirty changes to PR #40; branch experiment/h9-e1b-mmap-coresidency from master
- [x] Bring reusable E1 scripts + H9 doc onto branch
- [x] Create dev-docs (plan.md, context.md, tasks.md)

## Phase 2 — Plan + review
- [x] Plan-review loop (6 rounds, converged); addressed 5+5+3+3+3+3 findings
- [x] User approves plan

## Phase 3 — Implementation
- [x] Write `scripts/h9_e1b_coresidency.py` (calibrate + run B1/B2; reuses E1 collector/SSE)
- [x] Phase A: download Q4_K_M GGUF; sha 0d003f66 verified; disk-churn calibration PASS
- [x] Phase A: launch llama-server (mmap, --jinja); idle footprint measured
- [x] Phase A: idle ngl-0 = 17.91 GB phys_footprint BUT 4.01 GB resident (the crux finding)
- [~] Phase B: 8-min ngl-0 B1 under 8 GB ballast (clean, 14 samples); decode swap-limited.
      30-min/B2 deferred — needs quiesced machine (host contention, see context.md)
- [x] Phase C: 20-case harness on CPU = 18/20 (90%) PASS
- [x] Phase D: gates computed, logged to experiments.jsonl, verdict written

## Phase 4 — Code review
- [x] Commit WIP; code-review R1 (fixed requests dep, calibrate guard, p95, fd, mem_method, lints)
- [ ] Final-state code-review loop

## Phase 5 — Finalize
- [x] Final findings + verdict + E3 implication in context.md
- [ ] PR with Fixes #41, review round counts
- [ ] On merge: move dev-docs to dev/completed/
