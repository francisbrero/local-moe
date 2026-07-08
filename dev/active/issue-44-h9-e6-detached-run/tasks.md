# Tasks — Issue #44 (H9-E6 detached run)

## Phase 1: Setup
- [x] Read issue #44
- [x] Analyze reuse sources (E1b, E2, E4, harness, experiment_utils)
- [x] Create branch `experiment/issue-44-h9-e6-detached-run`
- [x] Create dev-docs (plan.md, context.md, tasks.md)

## Phase 2: Plan + review
- [x] Draft plan.md
- [x] Automated plan-review loop (converge on no material findings)
- [x] User approves plan

## Phase 3: Implementation
- [x] Preflight hard gate (< 19 GB abort, logs aborted record)
- [x] HOW TO RUN DETACHED banner + comment block
- [x] Phase A: E1b partial-offload decode (--e1b-ngl default 24, --cache-ram 512) + tool-call harness
- [x] Phase B Arm 1: E2 MLX kv4 sweep (reuse h9_e2_kv_workload)
- [x] Phase B Arm 2: E2 llama.cpp -ctk q8_0 vs f16 (NEW code)
- [x] Phase C: E4 idle MLX baseline (decode/prefill/phys + tool-call engine)
- [x] Phase D: h9_e6_SUMMARY overall verdict record
- [x] Optional launchd plist template
- [x] Self-test: preflight math, import resolution, ruff lint
- [x] Attempt run (expect preflight abort from Cursor session)

## Phase 4: Code review
- [ ] Automated code-review loop (converge)

## Phase 5: Finalize
- [ ] Commit with `Fixes #44` (or partial ref if numbers pending)
- [ ] PR with review round counts
- [ ] context.md final findings
