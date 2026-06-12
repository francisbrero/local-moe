# H9-E1 Baseline — Context / Checkpoint

**Issue:** #35 | **Branch:** `experiment/h9-e1-tier1-baseline`

## Current Phase
Phase 2 — plan drafted, entering automated plan review loop.

## Key Files
- `dev/active/issue-35-h9-e1-tier1-baseline/plan.md` — approach
- `dev/active/issue-35-h9-e1-tier1-baseline/tasks.md` — checklist
- `scripts/experiment_utils.py` — logging / RSS / vm_stat helpers (reuse)
- `scripts/h9_e1_baseline.py` — (to create) perf measurement
- `scripts/h9_e1_harness.py` — (to create) 20 tool-call cases + grader

## Environment Facts
- mlx_lm 0.31.1 installed; MLX GPU works. vllm-mlx NOT installed.
- Qwen3-30B-A3B-4bit already cached (no download).
- ~19 GB disk free (tight).

## Decisions
- Runtime: install vllm-mlx best-effort (per issue); measure baseline in-process via mlx_lm (user-directed).
- Baseline uses model-default top-k and no KV quant (clean reference for E2/E4).

## Review Rounds
- Plan review rounds: 6 (converged; rounds 1–5 surfaced real findings, all addressed; round 6 returned the round-5 findings verbatim with the reviewer confirming all are already fixed → convergence).
- Code review rounds: 0 (pending)

### Plan findings addressed
- R1: gated runtime should be served vllm-mlx not in-process; memory double-counting; missing TTFT; hypothesis/baseline/rollback sections.
- R2: no silent mlx_lm substitution; commit OpenAI tool_calls API shape; report headroom vs 19–21GB.
- R3: pin decode protocol (prompt len, max_new_tokens, N warm, streaming, transport); server process-tree memory; Phase A tool-call smoke test; no-tool case grading; fix misleading command comment.
- R4: HIGH stale 16GB→24GB hardware mismatch (corrected CLAUDE.md + plan note); guarantee non-empty gated result (engine tier); ppl/cache_hit_rate as null+reason; macOS phys_footprint fallback.
- R5: definitive served-vs-engine gating contract (engine insufficient to unblock E3/E5); concrete served-mode streaming timing fallback + null-with-reason; per-case semantic validators for tool-call gate; phys_footprint→RSS mem_method.

## Implementation Progress (Phase 3)

Plan APPROVED by user. Scripts written:
- `scripts/h9_e1_harness.py` — 20 cases (6 email triage, 4 calendar, 4 slack, 3 no-tool summarization,
  3 mixed multi-arg), semantic per-case validators, served+engine adapters, lenient-recovery parser.
- `scripts/h9_e1_baseline.py` — pinned decode protocol, macOS phys_footprint collector (proc_pid_rusage
  RUSAGE_INFO_V2), process-tree memory sampler, served (SSE) + engine (generate_step) timing.

### Phase A findings (vllm-mlx 0.3.0)
- **vllm-mlx 0.3.0 installed successfully** — full serving engine: `serve` (OpenAI-compatible),
  `--enable-auto-tool-choice`, `--tool-call-parser hermes` (Qwen3), `--enable-metrics`, kv-cache quant flags.
  Served tier is VIABLE → gate_tier=served is achievable.
- Server loads the model lazily on first request; tool parser `hermes` initializes correctly.

### KEY MEASURED FINDING (preliminary, served tier)
- **Idle (model-loaded) phys_footprint = 16.48–16.55 GB** for the server process tree.
  Standard RSS showed only ~0.37 GB — **confirms the plan's core point: RSS massively undercounts
  MLX wired/Metal memory; phys_footprint (proc_pid_rusage) is the correct gate metric.**
- At 16.5 GB resident on this 24GB machine (other apps ~2-3GB + OS), **available memory dropped to
  1.3–1.6 GB**. Under this pressure, **decode was pathologically slow** — a cold 32-token generation
  did not complete within minutes; the server logged "first chunk arrived 0.0s" then stalled (heavy
  paging / GPU memory contention). This is a real, important baseline result: the 30B-A3B 4-bit model
  leaves almost no headroom on 24GB, and decode degrades badly under that pressure.
- Implication for gates: the ≤17 GB footprint gate is borderline (16.5 GB idle, will exceed at 16K
  context once KV cache is added). The ≥20 tok/s decode gate is at risk under memory contention.

### ROOT CAUSE of slow decode: memory oversubscription / swapping (confirmed)
- Both served (vllm-mlx) and engine (mlx_lm) tiers loaded the model to **16.5 GB phys_footprint**.
- During decode, observed `vm.swapusage used = 15.9 GB`, system free 12%, worker CPU ~10% (blocked on I/O).
- Mechanism: 16.5 GB model + competing apps (Cursor, Xcode, Chrome, multiple Claude procs) exceed
  physical RAM → macOS pages the model in/out → decode throughput collapses (cold 32-tok gen did not
  finish in minutes).
- This is NOT a model/runtime defect; it's the 24GB headroom reality. The issue's intended condition
  is "alongside browser + Slack" (light), but the dev machine also runs Cursor+Xcode (heavy).
- **Honest baseline requires a clean-ish machine.** Re-measuring with memory freed (>=10GB avail).
- This finding itself partially answers the ≤17GB gate: idle model footprint 16.5 GB already leaves
  minimal headroom; with heavy apps the system swaps. Decode ≥20 tok/s is only achievable with
  adequate free RAM.

### Open issue being resolved
- Need clean decode/prefill numbers. HTTP-client timing stalled under memory pressure.
- Plan: use vllm-mlx's built-in `bench` (authoritative, single model copy) after freeing memory,
  AND/OR re-measure once nothing else competes for RAM. Waiting for available memory to recover >=5GB.

## FINAL FINDINGS (Phase 3 complete; perf gates memory-blocked)

User chose "measure under contention." Under the dev machine's normal app load, the result is:

| Gate | Target | Measured | Verdict |
|---|---|---|---|
| Idle model footprint | (informational) | **16.55 GB** (phys_footprint) | — |
| Peak footprint @16K | ≤ 17 GB | ≥16.55 GB idle, exceeds at 16K ctx | **borderline / likely FAIL at 16K** |
| Decode | ≥ 20 tok/s | **unmeasurable** — swap-limited | **FAIL (under contention)** |
| Prefill 8K/16K | ≥ 300 tok/s | unmeasurable | BLOCKED |
| Tool-call success | ≥ 90% | not run (same load bottleneck) | BLOCKED |

### What is solid
- **vllm-mlx 0.3.0 installs and serves Qwen3-30B-A3B** with OpenAI-compatible endpoint, tool-calling
  (`--enable-auto-tool-choice --tool-call-parser hermes`), metrics. Served tier is viable.
- **Idle model footprint = 16.55 GB** via phys_footprint (proc_pid_rusage). RSS reports only ~0.37 GB
  — confirms the plan's thesis that RSS is the wrong metric for MLX; phys_footprint is authoritative.
- Tool parser `hermes` initializes; a trivial tool-call smoke request returned 200 (server-side).

### Root cause of the block (reproducible across both runtimes)
- 16.55 GB model + competing apps (Cursor, Xcode, Chrome, multiple Claude procs) > physical RAM.
- System enters sustained swap (vm.swapusage used 14–19 GB, free ~12%); model **load itself** does
  not complete to usable speed (~20 min, worker 3–36% CPU, I/O-bound). Decode never reaches steady state.
- This is the issue's own "alongside browser+Slack" requirement failing in practice, made worse by the
  heavier dev apps present. It is a headroom problem, not a model/runtime defect.

### To get true gate numbers (deferred)
- Re-run on a **quiesced machine** (close Cursor/Xcode/Chrome). With ~17 GB free the model fits with
  a few GB headroom and decode should reach the MLX-typical 30–50 tok/s cited in the H9 doc.
- The scripts are ready: `uv run python scripts/h9_e1_baseline.py --runtime engine` and
  `--runtime served --server-pid <pid>`; harness via `scripts/h9_e1_harness.py --runtime auto`.

## Next Steps
1. Commit scripts + docs (WIP).
2. Phase 4 code-review loop on the scripts.
3. PR documenting: tooling built, footprint finding, swap-limited blocker, quiesced re-run needed.
