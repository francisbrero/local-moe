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
- [~] Phase A: download Q4_K_M GGUF (running, bg); repo=Qwen/Qwen3-30B-A3B-GGUF, 18.56 GB, sha 0d003f66
- [ ] Phase A: launch llama-server (mmap, --jinja, --metrics); tool-call smoke test; idle footprint
- [ ] Phase A: measure -ngl 999 AND CPU/partial config idle footprint (the crux)
- [ ] Phase B: 30-min co-residency under 8 GB ballast, sample phys_footprint + vm_stat every 30s
- [ ] Phase C: run 20-case harness (served, port 8124); record success rate
- [ ] Phase D: compute gates, log to experiments.jsonl (phys_footprint), write verdict

## Phase 4 — Code review
- [ ] Commit WIP; code-review loop (max 10 rounds), fix material findings

## Phase 5 — Finalize
- [ ] Final findings + review stats in context.md
- [ ] PR with Fixes #41, review round counts
- [ ] On merge: move dev-docs to dev/completed/
