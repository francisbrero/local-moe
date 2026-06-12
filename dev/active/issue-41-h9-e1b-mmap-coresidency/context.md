# H9-E1b — Context / Checkpoint

**Issue:** #41 | **Branch:** `experiment/h9-e1b-mmap-coresidency`

## Current Phase
Phase 2 — plan drafted (`plan.md`), entering automated plan-review loop.

## The question
Does Qwen3-30B-A3B in **llama.cpp mmap mode** survive co-residency under 8 GB ballast
where **MLX-wired thrashed** (E1/#35: 16.55 GB wired, swap-collapse, 4× load failures)?
PASS → 30B viable as concurrent daily-driver. FAIL → concurrent tier = ≤ 4B only.

## Key files
- `plan.md` — approach, gates, risks
- `tasks.md` — checklist
- `scripts/h9_e1_baseline.py` — REUSE: phys_footprint collector + process-tree sampler
- `scripts/h9_e1_harness.py` — REUSE: 20-case tool-call harness (served = OpenAI HTTP)
- `scripts/experiment_utils.py` — REUSE: create_memory_pressure, vm_stat, log_experiment
- `scripts/h9_e1b_coresidency.py` — TO CREATE: llama-server driver + 30-min sampling loop

## Environment facts (verified)
- llama-server: `/Users/francis/.docker/bin/inference/llama-server` (build 65ef50a, arm64).
  Has --mmap/--no-mmap, --mlock, -ngl, --jinja, --metrics, --port. OpenAI-compatible API.
- NO Qwen3-30B-A3B GGUF cached (only MLX 4-bit). Must download Q4_K_M (~18 GB). ~63 GB free.
- E1 scripts + H9 doc checked out from #35 branch (PR #40 unmerged; not yet on master).

## Gates (all 6 must pass — plan-review R2/R5)
- Decode ≥ 12 tok/s sustained under 8 GB ballast
- p50 TTFT ≤ 10 s under 8 GB ballast
- Pageouts < 200 MB/hr over 30 min
- phys_footprint p95 AND max ≤ 8 GB over the window
- Disk-read churn < 50 MB/s sustained (ri_diskio_bytesread, per-process; calibration step required)
- Tool-call success ≥ 90% semantic (20-case harness, matching E1 threshold)

vm_stat pageins are logged as a diagnostic cross-check but are NOT a gate (system-wide proxy, not attributable to the server process).

## Verdict classes (plan-review R2)
- **FULL PASS**: all 6 gates on B1 (30-min sustain) AND B2 load-success + ≥ 5-min sustain.
- **CONDITIONAL PASS**: all 6 gates on B1, but B2 fails to load/sustain. 30B viable post-load, not guaranteed under pre-occupied machine. E3 may proceed with caveat noted.
- **FAIL**: any gate fails on B1, or both B1+B2 fail. Concurrent tier = ≤ 4B only.

## Crux / central risk
With Metal offload (`-ngl 999`, the normal Mac path) weights may be wired into unified
memory — negating mmap, reproducing MLX's failure. Mitigation: measure BOTH ngl-999 and a
CPU/partial config; verdict uses best config meeting decode gate. If none passes → honest FAIL.

## Decisions
- Branch made from clean master; #35's uncommitted research-doc edits committed to PR #40 first.
- Reusable E1 scripts brought onto this branch via `git checkout <#35> -- <files>`.

## Review Rounds
- Plan review rounds: 6 (converged; round 6 returned material_findings:false after self-applied fixes).
  - R1: ballast API mismatch (create_memory_pressure leaves N avail, not allocates N) → fixed-8GB ballast; pin llama.cpp ctx/batch/KV/prompt; -ngl 0 mandatory; buffer-size logging; runtime+format (not single-var) framing.
  - R2: cache-state declaration (warm) + warmup protocol; mmap page-cache blind spot → log ri_resident_size/diskio + swap/compressor, condition PASS on low churn; required benchmark fields; streaming TTFT; Q4_K_M mandatory for verdict.
  - R3: numeric disk-churn gate (<50 MB/s) + pagein bar; phys_footprint gate = p95 AND max; partial-offload mandatory if extremes fail.
  - R4: exact per-process measurement sources (ri_diskio_bytesread); B1/B2 load-order variants (load-under-ballast); ballast residency under macOS compressor (random fill + periodic re-touch).
  - R5: pageins demoted to diagnostic (not gate); gate count unified; B2 verdict classes (FULL/CONDITIONAL/FAIL).
  - R6: TTFT hard gate (≤10s); tool-call ≥90% hard gate; ri_diskio_bytesread calibration preflight; bounded partial-offload search; verdict-wording fix. Converged.
- Total plan findings addressed: 5+5+3(+1 low)+3(+2 low)+3+3(+2 low) across R1–R6 (all high/medium addressed; several lows too).
- Code review rounds: 0
- Post-convergence consistency fix: corrected stale "all 4 gates" → "all 6 gates" in plan.md (3 sites) after R6 self-edits added gates 5–6.

## Next Steps
1. Run plan-review loop, address findings, get user approval.
2. Phase A: download Q4_K_M GGUF, launch llama-server, smoke-test tool call, measure idle footprint.
3. Phase B: 30-min co-residency under 8 GB ballast, sample every 30 s.
4. Phase C: 20-case harness. Phase D: verdict + experiments.jsonl + PR.
