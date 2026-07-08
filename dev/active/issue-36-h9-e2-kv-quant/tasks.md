# H9-E2 — Tasks

## Phase 0 — Setup
- [x] Validate issue #36, read H9 doc + rescope comment
- [x] Create branch `experiment/h9-e2-kv-quant-workload-context`
- [x] Scaffold `dev/active/issue-36-h9-e2-kv-quant/{plan,context,tasks}.md`

## Phase 1 — Plan + review
- [x] Draft plan.md
- [x] Automated plan-review loop (6 rounds, 12 findings addressed)
- [x] User approval

## Phase 2 — Implementation (Arm a, primary)
- [x] `scripts/h9_e2_kv_workload.py` — synthetic transcript generator (10K/12K/15K, seeded)
- [x] KV-cache sizing via direct `.nbytes` (gate) + analytic cross-check (3.56× verified)
- [x] Quality: summary-only teacher-forced ΔPPL + ROUGE-L(kv4 vs kv16), write summaries to logs/
- [ ] Phase A smoke test **on quiesced machine** (blocked: 10.9 GB free < ~18 GB needed)
- [ ] Phase B sweep: 3 lengths × {kv16, kv4} × 5 samples (quiesced)
- [ ] Phase C gates: ΔPPL, ROUGE-L, effective-context-per-GB ≥ 1.5×
- [ ] Phase D log to experiments.jsonl + verdict
- [x] Arm (b): documented as deferred/out-of-scope (E1b host blocker)

## Phase 3 — Code review
- [x] Commit WIP
- [x] Automated code-review loop (3 rounds; converged; 8 findings incl. 1 high fixed)

## Phase 4 — Finalize
- [x] Update context.md (findings + review stats)
- [x] Commits reference #36
- [x] `gh pr create`
- [ ] Run the sweep on a quiesced machine; record measured verdict (blocked: 10.9 GB free)
