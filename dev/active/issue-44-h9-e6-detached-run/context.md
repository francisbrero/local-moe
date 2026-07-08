# Context — Issue #44 (H9-E6 detached run)

## Current phase
Phase 3/4 — script implemented + self-tested; entering code-review loop.

## Implementation status
- `scripts/h9_e6_detached_run.py` — full bundled script, lint-clean, all reused symbols resolve.
  - Preflight verified: aborts with exit 2 at <19 GB free, logs `h9_e6_preflight` aborted record.
  - Orchestration plumbing (force/skip/cooldown/SUMMARY) verified without model load.
  - Vacuous-true guard added (all-skipped → overall gate False).
- `scripts/h9_e6.plist` — optional launchd template (2 AM, not installed).
- E4 + E2-MLX delegate to standalone scripts as subprocesses (clean per-phase MLX teardown).
- E1b + E2-llama.cpp orchestrated in-process at E1b primitives (Ballast/tree_counters/wait_healthy).
- E2 llama.cpp q8_0-vs-f16 arm is the only net-new code; quality scorer probes input-token
  logprobs (expected unavailable → quality sub-gate None, pageout/mem numbers still recorded).

## Run reality from THIS session
Machine has ~4 GB free (Cursor/agent/Chrome live) → preflight aborts by design. The genuine
`h9_e6_preflight aborted` record is in experiments.jsonl. Real numbers require a detached run
(quit apps, run from bare terminal / the launchd plist). 3 duplicate test-artifact records were
removed from experiments.jsonl; 1 genuine aborted-preflight record kept.

## What this is
A single bundled detached run script (`scripts/h9_e6_detached_run.py`) that produces the
measured numbers three prior experiments (E1b/#41, E2/#36, E4/#38) could never get because the
orchestrating IDE session was itself competing for the 24 GB. Hard preflight gate (≥19 GB free)
enforces detachment.

## Key decisions
- User: **build + attempt run now** (preflight abort from Cursor is the expected/documented outcome).
- User: E1b `-ngl` **parameterized, default 24 (50% of 48 layers)**, `--cache-ram` trimmed.
- The **llama.cpp `-ctk q8_0` KV arm does not exist** in the E2 driver — it's the one net-new
  piece of code. Everything else is reuse.

## Reuse anchors (verified via subagent recon)
- Logging: `experiment_utils.log_experiment` (schema: experiment_name/phase/status/config/results/env/meta).
- rusage: `h9_e1_baseline._RUsageInfoV2/_libc/_RUSAGE_INFO_V2/tree_pids`; `h9_e1b_coresidency.tree_counters`.
- Timing: `h9_e1_baseline.served_generate_timed`.
- E1b: `h9_e1b_coresidency.start_server/wait_healthy/run/Ballast/_compute_gates` (llama-server @ port 8124).
- E2 MLX: `h9_e2_kv_workload.preflight/cmd_run/_summarize_once/teacher_forced_nll` (MODEL mlx-community/Qwen3-30B-A3B-4bit).
- E4/E1 MLX: `h9_e1_baseline.run_engine/_assemble` (decode/prefill/phys peak).
- Harness: `h9_e1_harness` 20 cases; `--runtime served --port 8124` (E1b) / `--runtime engine` (E4).
- Model ids: MLX `mlx-community/Qwen3-30B-A3B-4bit`; GGUF `models/gguf/Qwen3-30B-A3B-Q4_K_M.gguf`.

## Gotchas
- Full Metal offload (-ngl 999) is GPU-OOM (18186 MiB budget) — MUST partial-offload + trim cache-ram.
- `create_memory_pressure` allocates DOWN TO target, not target GB — use E1b `Ballast` (fixed 8 GB) instead.
- KV quant in llama.cpp needs `-fa` (flash attention) alongside `-ctk q8_0`.
- See [[h9-machine-contention-blocker]] memory.

## Review rounds
- Plan review rounds: 4 (converged, MATERIAL_FINDINGS:false)
  - R1: E1b teardown/metric, inter-phase cleanup, E4 ordering, E2-arm2 quality gate, llama.cpp fields
  - R2: E1b unified-mem budget gate, unavailable llama-perplexity, empty corpus, abs+rel pageout, mlx_touched, doc step
  - R3: input-token logprob coverage, distinct ports, execution-order clarity, start_server copy-not-call
  - R4: clean
- Code review rounds: 2 (converged, MATERIAL_FINDINGS:false)
  - R1 (3 medium, all fixed): E4 tool-call parse/gate, tokenizer-only corpus (no 30B weights), E2-llama honors lowmem gate
  - R2 (2 low, fixed anyway): dead corpus[0] guard, Popen file-handle leak in _run_e2_llama_arm
- Total findings addressed: 9 plan + 5 code = 14

## FINAL STATUS
Deliverables complete:
- `scripts/h9_e6_detached_run.py` — bundled E1b+E2+E4 detached run, lint-clean, self-tested.
- `scripts/h9_e6.plist` — optional launchd overnight template.
Verified from this session: preflight aborts (exit 2) at ~4 GB free, logs h9_e6_preflight aborted.
Real measured numbers are pending the user's DETACHED run (quit IDE/Chrome, run from bare
terminal). This is the designed outcome — the whole point of #44 is that numbers cannot be
obtained from the contended orchestrating session. Follow-ups (#36/#37/#38/PR#43 updates, moving
dev-docs to completed/) happen once the detached run produces numbers.

## Next steps
1. Plan-review loop.
2. User approval.
3. Implement script.
