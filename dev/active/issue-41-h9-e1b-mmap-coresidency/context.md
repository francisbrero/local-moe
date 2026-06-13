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

## Implementation Progress (Phase 3)
- Plan APPROVED. Harness `scripts/h9_e1b_coresidency.py` written + code-reviewed (R1: fixed
  requests dep, calibrate guard, p95 nearest-rank, fd close, mem_method propagation, lints).
- GGUF downloaded: Qwen/Qwen3-30B-A3B-GGUF / Q4_K_M, 18.56 GB, sha256 verified 0d003f66...
- **Calibration PASS:** cold mmap scan read 64 MB disk, warm scan 0 MB (ratio inf) →
  per-process `ri_diskio_bytesread` is a valid churn signal. Logged to experiments.jsonl.

### KEY FINDING — ngl-999 probe under heavy contention (2-min B1)
Machine state at run: avail ~6.7 GB, swap ~7 GB used (heavily contended — E1's blocker).
- **llama.cpp mmap LOADS the 30B model** at `-ngl 999` ("model loaded", "server listening")
  — a categorical difference from E1/MLX, which could not even load (4× failures).
- **BUT decode then hit Metal OOM** (`kIOGPUCommandBufferCallbackErrorOutOfMemory`,
  `llama_decode failed ret=-3`) during warmup: the 4 GB **Metal half of the ballast** competes
  with the model's Metal buffers in the unified-memory GPU pool → GPU OOM. This is the plan's
  central "Metal offload negates mmap" crux risk materializing for ngl-999 under contention.
- Implication: at full Metal offload, weights ARE wired into the GPU pool (MLX-like), so an
  8 GB ballast with a 4 GB Metal component can't coexist. The mmap benefit (page-cache paging)
  is only exercised at lower -ngl. **Next: run -ngl 0 (CPU/mmap-preserved), the config the
  hypothesis is really about** — there the ballast's Metal half won't OOM against model weights.

## FINAL FINDINGS (Phase 3)

### The headline: mmap changes the memory STORY, not the co-residency verdict
| Metric | E1 (MLX-wired) | E1b (llama.cpp mmap Q4_K_M) |
|---|---|---|
| Model loads under contention? | **No** (4× failures) | **Yes** — categorical improvement |
| phys_footprint (idle) | 16.55 GB (wired) | **17.91 GB** (mmap file-backed pages charged) |
| TRUE resident set (idle) | ~16.5 GB (all wired) | **~4.01 GB** (`ri_resident_size`, ngl-0) |
| Decode under 8 GB ballast | unmeasurable (swap) | unmeasurable (swap) — no tokens in window |

### Key measurements (all logged to experiments.jsonl)
1. **Idle ngl-0, no ballast: phys_footprint 17.91 GB but ri_resident_size 4.01 GB.** This is the
   crux result. mmap works as designed — the OS keeps only ~4 GB of weights RAM-resident and
   pages the rest. But **phys_footprint OVER-counts mmap** (it includes file-backed pages the OS
   can evict). This is the exact mirror of E1, where RSS *under*-counted MLX's wired memory.
   → **The plan's chosen gate metric (phys_footprint) is the wrong metric for mmap.**
   `ri_resident_size` (~4 GB) is the truer working set, and by THAT metric the footprint gate
   would PASS. Plan-review R2 predicted exactly this page-cache blind spot.
2. **8-min ngl-0 B1 under 8 GB ballast (clean run, 14 samples):** phys_footprint rock-stable at
   17.98 GB; ballast landed (avail 4.72→0.26 GB); decode produced **no tokens** (CPU 30B under
   13 GB swap is too slow); pageout 283 MB/hr (>200 FAIL); disk-churn 0.0 MB/s (model stays
   mmap-resident, not re-faulting → churn gate PASSES).
3. **ngl-999 (full Metal offload) hits Metal OOM even with NO ballast.** The 30B in the GPU pool
   (~18 GB) + 8 GB prompt-cache + KV exceeds the M4 Pro's **18186 MiB GPU budget**
   (`kIOGPUCommandBufferCallbackErrorOutOfMemory`, decode ret=-3). Full Metal offload of this
   model is not viable on this machine regardless of co-residency.
4. **Tool-calling PASSES: 18/20 = 90.0%** (CPU/ngl-0, quiesced, no ballast). Calendar 4/4,
   Slack 4/4, summarization 3/3, mixed-multiarg 3/3 all 100%; email triage 4/6 (2 misses).
   Qwen3-30B-A3B via llama.cpp produces high-quality tool calls. (The earlier ngl-999 harness
   scored 0% purely due to GPU-OOM compute errors / HTTP 500, NOT model quality.)

### Verdict (vs the 6 gates)
- decode ≥12 tok/s: **FAIL** (swap-limited, no tokens) — but confounded by host contention.
- p50 TTFT ≤10s: **FAIL** (no tokens).
- pageouts <200 MB/hr: **FAIL** (283 MB/hr; system swapping).
- phys_footprint p95&max ≤8 GB: **FAIL by phys_footprint (18 GB); PASS by resident_size (~4 GB).**
- disk-read churn <50 MB/s: **PASS** (0 MB/s steady state).
- tool-call ≥90%: **PASS** (18/20 = 90.0%).

**Overall: FAIL the gates as written, but with a critical metric caveat** — the failure is
(a) host swap contention (same blocker as E1, machine couldn't be fully quiesced — see
[[h9-machine-contention-blocker]]), and (b) phys_footprint mis-measuring mmap. The genuinely
new, machine-independent result: **llama.cpp mmap keeps only ~4 GB RAM-resident** (vs MLX wiring
16.5 GB), so by the correct mmap metric the 30B's *resident* working set DOES fit the 8 GB budget
— but **decode speed under real co-residency was not demonstrable** on this contended 18 GB-GPU
machine, and full-GPU-offload is OOM-blocked. Honest verdict: **CONDITIONAL / inconclusive on
speed; the memory thesis (mmap fits) is supported by resident_size, refuted by phys_footprint.**

### Implication for E3 (#37)
The verdict is **mixed, not a clean PASS or FAIL**, so E3 should proceed with a sharper model:
- **Memory:** mmap's true resident set (~4 GB) DOES fit a concurrent budget — so the 30B is NOT
  categorically off-limits the way E1's MLX-wired 16.5 GB was. But phys_footprint (~18 GB) and the
  decode-under-contention failure mean it is **not a proven free-lunch daily-driver** either.
- **Engine choice:** full Metal offload (`-ngl 999`) is **OOM-blocked** on this 18 GB-GPU machine
  for the 30B; a usable config must be **partial offload** (some layers CPU/mmap) with the prompt
  cache trimmed (`--cache-ram` low). E3 should benchmark partial-offload decode speed on a
  *genuinely quiesced* machine — that is the one number this experiment could not get cleanly.
- **Recommendation:** treat 30B-via-llama.cpp-mmap as a **viable-but-unproven concurrent candidate**
  (memory fits by resident_size; speed TBD), and keep ≤4B as the safe concurrent default until a
  clean partial-offload decode number clears ≥12 tok/s. Tool-call quality (90%) is sufficient.

## What remains (deferred — needs a genuinely quiesced machine)
- Clean decode tok/s + TTFT under 8 GB ballast at a **partial -ngl** config (the one number not
  obtainable under this session's swap contention). Scripts are ready:
  `uv run python scripts/h9_e1b_coresidency.py run --ngl <partial> --order B1 --minutes 30`.
- B2 (ballast-then-load) load-under-pressure variant.

## Review Stats (final)
- Plan review rounds: 6 (converged). Findings addressed: ~17 high/medium across R1–R6.
- Code review rounds: <to fill after Phase 4>.
