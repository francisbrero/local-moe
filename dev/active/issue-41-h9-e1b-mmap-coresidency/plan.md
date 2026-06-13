# H9-E1b — llama.cpp mmap co-residency probe

**Issue:** #41 | **Branch:** `experiment/h9-e1b-mmap-coresidency` | **Depends on:** #35 (E1, closed/PR #40)

## Hypothesis

The **same Qwen3-30B-A3B model in llama.cpp mmap mode** survives co-residency under
8 GB competing ballast where **MLX-wired thrashed**. E1 (#35) proved MLX wires the full
**16.55 GB phys_footprint** and swap-collapses on 24 GB with any real working set
(load failure reproduced 4×). The mmap thesis: llama.cpp memory-maps the GGUF and lets
the **OS page experts via the page cache** rather than wiring everything, so resident
working set stays far below the file size and the model coexists with an 8 GB ballast.

This experiment **decides the shape of the concurrent tier**:
- **PASS** → 30B is a viable daily-driver concurrent model (mmap mode).
- **FAIL** → concurrent tier defaults to ≤ 4B-class; 30B is off-hours/batch only.

Either outcome is a coherent, publishable result (negative results are valuable).

## Baseline (the thing we are comparing against)

E1 / MLX-wired, on this machine, from `experiments.jsonl` + #35 context:
- Idle model footprint **16.55 GB** phys_footprint (RSS reported only ~0.37 GB — RSS is
  unreliable for MLX/Metal; **phys_footprint is the authoritative gate metric**).
- Under any competing working set: sustained swap (vm.swapusage 14–19 GB), decode
  swap-limited / unmeasurable, load itself never reaches usable speed.

E1b changes the **runtime and weight format** (MLX 4-bit, wired → llama.cpp GGUF Q4_K_M,
mmap). Same **model family**, same **hardware**, same **phys_footprint metric**, same
**20-case tool-call harness**, same **8 GB ballast**. Note both the runtime *and* the
quantization implementation change (MLX-4bit vs GGUF Q4_K_M) — so we log the GGUF quant
metadata + file SHA and interpret the result as "same model family, runtime+format changed,"
not a pure single-variable swap. (Pinned to address plan-review R1.)

### Ballast: exactly 8 GB allocated (NOT `create_memory_pressure`)
**Critical correction (plan-review R1):** `experiment_utils.create_memory_pressure(t)` leaves
`t` GB *available* — it does NOT allocate `t` GB of pressure. The issue requires an **8 GB
competing working set**, so we allocate a fixed ballast directly:
`allocate_cpu_ballast(4.0)` + `allocate_mlx_ballast(4.0)` = 8 GB (4 GB CPU + 4 GB Metal,
both force-touched/eval'd), holding references for the entire run. Log ballast
phys_footprint and `get_available_memory_gb()` before/after allocation to confirm the 8 GB
landed. Rationale for the CPU/Metal split: mirrors H8b and exercises both the unified-memory
pool (Metal, competing with model weights) and host RAM (CPU, competing with page cache).

**Ballast must stay genuinely resident (plan-review R4-medium):** under macOS memory pressure
the compressor will silently compress/purge anonymous pages that aren't recently touched —
which would shrink the effective competing working set and invalidate the test. Mitigations:
(a) fill the CPU ballast with **non-compressible random bytes** (not zeros — zero pages
compress to nothing), (b) **periodically re-touch** the ballast (read/write a byte per page
every ~30 s, in the driver's sample loop) so it stays hot and resident, (c) log ballast-PID
phys_footprint + compressor/swap state each sample to *prove* the 8 GB stayed resident across
the full 30 min. `experiment_utils.allocate_cpu_ballast` currently writes `0xFF` per page
(non-zero, good) but the same constant; the driver fills with a varying/random pattern and
adds the periodic re-touch — note this as a small extension of the helper.

## Environment facts (verified)

- `llama-server` binary present: `/Users/francis/.docker/bin/inference/llama-server`
  (version 1, build 65ef50a, AppleClang, Darwin arm64). Supports `--mmap/--no-mmap`,
  `--mlock`, `-ngl`, `--jinja` (tool calls), `--metrics`, `--port`. OpenAI-compatible
  `/v1/chat/completions` → the existing E1 served adapter drives it unchanged.
- **No Qwen3-30B-A3B GGUF cached** — only the MLX 4-bit copy. Must download
  **Q4_K_M GGUF** (~18 GB). Disk: ~63 GB free → fits.
- Reusable from #35 (checked out onto this branch):
  - `scripts/h9_e1_baseline.py` — phys_footprint collector (`proc_pid_rusage`
    `RUSAGE_INFO_V2`), process-tree memory sampler, served-tier SSE timing.
  - `scripts/h9_e1_harness.py` — 20-case tool-call harness (OpenAI tool API; served
    adapter hits `/v1/chat/completions`), semantic per-case validators, lenient parser.
  - `scripts/experiment_utils.py` — `create_memory_pressure`, `get_vm_stat`,
    `vm_stat_delta`, `log_experiment`, `get_environment_info`.

## Gates (issue #41 — all must pass for PASS verdict)

| Gate | Target |
|---|---|
| Decode tok/s (sustained, not burst) | **≥ 12** under 8 GB ballast |
| p50 first-token latency (TTFT) | **≤ 10 s** under 8 GB ballast |
| Pageouts over the 30-min window | **< 200 MB/hr** |
| phys_footprint (resident working set) | **p95 AND max ≤ 8 GB** over the window |
| Disk-read churn (steady-state) | **< 50 MB/s** sustained `ri_diskio_bytesread` |
| Tool-call success (20-case harness) | **≥ 90%** semantic success |

**Pageins (vm_stat) — diagnostic only, NOT a hard gate (plan-review R2):** system-wide
`vm_stat` pageins are not attributable to llama.cpp alone; unrelated processes can trivially
exceed any threshold, producing false failures. `ri_diskio_bytesread` (per-process, above) is
the authoritative per-process mmap churn signal. Pageins are logged alongside it as a
cross-check: if vm_stat pageins spike but per-process disk reads stay low, the churn is some
other process, not the server. Logged always; never a gate.

**phys_footprint gate semantics (plan-review R3-low):** the gate is the **p95 and max**
process-tree phys_footprint over the sustained window (median is reported separately but is
not the gate) — this catches transient Metal/KV/scratch spikes that a median would hide.
Child processes included; ballast footprint recorded separately.

**Numeric churn gate (plan-review R3-high):** "low churn" is now quantified. Because mmap
file-backed pages are mostly *uncharged* to phys_footprint, a config could meet `≤ 8 GB`
while continuously faulting GGUF pages from SSD — that is the mmap analogue of E1's
swap-collapse and is a **FAIL**. The thresholds, set before running:
- **steady-state disk read < 50 MB/s** (per-process `ri_diskio_bytesread` delta / elapsed,
  measured over the back-half of the 30-min window to exclude load/warmup). The full 30B
  Q4_K_M is ~18 GB; sustained reads near SSD bandwidth (GB/s) = active thrash. 50 MB/s is a
  generous "occasional fault, not churning" ceiling (≈ the model re-read every ~6 min, well
  below daily-driver-usable).
**PASS requires all 6 gates in the table above** (decode tok/s + p50 TTFT + pageouts +
phys_footprint + disk-read churn + tool-call success). phys_footprint alone is insufficient;
pageout + disk-read-churn together are the real co-residency test. vm_stat pageins are logged
as a diagnostic cross-check (see note above) but are not themselves a gate.

**TTFT gate (plan-review R5-medium):** p50 first-token latency is a hard gate at **≤ 10 s**
under 8 GB ballast. mmap/page-cache pressure can spike prefill latency even when steady-state
decode tok/s looks fine — a 60 s first token is not a daily-driver. If p50 TTFT > 10 s the
config is a FAIL regardless of decode rate. p95 TTFT is logged separately for diagnostic
purposes. Rationale: E1's interactive agentic-ops context (email triage, calendar, Slack) needs
a responsive first token; ≤ 10 s is a generous threshold consistent with a server-side call.

**Tool-call quality gate (plan-review R5-medium):** ≥ 90% semantic success on the 20-case
harness is a hard gate (matching E1's threshold). A config that passes memory + speed gates
but fails the quality gate is a FAIL for daily-driver purposes — both the runtime and
quantization change, so quality must be re-confirmed. Structural vs recovered breakdown is
still logged separately.

## Pinned configuration (plan-review R1 — gates are not reproducible without this)

All runs use these fixed parameters; every value is logged into the `config` block of each
`experiments.jsonl` record. A PASS is only valid at the **daily-driver context size**, so a
tiny `--ctx-size` cannot trivially pass the memory gate.

| Param | Value | Why |
|---|---|---|
| `--ctx-size` | **8192** | Realistic agentic-ops context (email/Slack/calendar threads). PASS gate is conditioned on this size; a smaller ctx is not gate-eligible. |
| `--batch-size` / `--ubatch-size` | **2048 / 512** (llama.cpp defaults) | Pinned for reproducibility; default scratch sizing. |
| `--parallel` (slots) | **1** | Single concurrent request (daily-driver, not a server fleet). KV cache sized for 1 slot. |
| KV cache type | **f16** (default, no quant) | Clean reference; matches E1 "no KV quant" baseline. KV-quant is E4's job. |
| Decode prompt | **fixed ~512-token prompt**, `max_tokens=256`, **streaming (SSE)**; the harness uses its own per-case prompts | Pinned so sustained tok/s is comparable run-to-run and to E1's protocol. **Streaming so p50 TTFT is measurable** (plan-review R4): TTFT = time to first SSE chunk; decode tok/s = generated-tokens / (total − TTFT), excluding the first token. |
| Sampling | greedy (`temperature 0`) for the decode-rate probe | Deterministic, removes sampler variance from tok/s. |
| `-ngl` | **999** AND **0** (both mandatory) | The crux — see Phase A.2. Both are measured every run, not conditionally. |
| mmap | **enabled** (no `--no-mmap`), `--mlock` **off** | The whole point: let the OS page via the page cache. |

Startup-log capture: record llama.cpp's printed **CPU/Metal model buffer sizes** and
**KV cache size** from the load banner — this is the ground truth for where the weights
landed (page cache vs wired Metal buffer) and directly answers the mmap-vs-Metal question.

Quality control: tool-call success is the primary quality signal. Additionally run a
**lightweight perplexity/quality probe** — a small fixed prompt set scored for coherence —
or, if a clean perplexity number isn't cheap via llama-server, mark `perplexity=null` with
reason and rely on the 20-case harness as the quality gate (matches E1's `ppl=null+reason`).
GGUF page residency / cache-hit-rate is marked **N/A** (no clean API) with reason logged.

### mmap observability — phys_footprint alone is not enough (plan-review R2)
The central interpretation caveat for an mmap experiment: file-backed GGUF pages live in the
**OS page cache, outside the server's charged `phys_footprint`**. So a run could satisfy the
`phys_footprint ≤ 8 GB` gate while still leaning on large cache residency or heavy file-backed
churn (which would evict the ballast / other resident data — exactly the daily-driver pain we
care about). We keep `phys_footprint` as the **gate metric** (E1 comparability) but log a full
pressure picture and condition the PASS on it. **Exact measurement sources (plan-review
R4-high)** — each metric names its collector so churn is verifiable, not hand-waved:
- **phys_footprint + `ri_resident_size`**: per-PID `proc_pid_rusage(RUSAGE_INFO_V2)` summed
  over the server **process tree** (the collector already in `h9_e1_baseline.py:80-129`).
- **disk-read churn**: per-PID `ri_diskio_bytesread` from the *same* RUSAGE_INFO_V2 struct
  (add the field to the ctypes struct), summed over the process tree, delta/elapsed → MB/s.
  This is the authoritative per-process mmap-fault signal — NOT a system-wide proxy.
- **pageins/pageouts**: system-wide `vm_stat` (via `experiment_utils.get_vm_stat` /
  `vm_stat_delta`). Explicitly treated as a **noisy system-wide proxy** and read *alongside*
  the per-process `ri_diskio_bytesread` (the precise signal) — if vm_stat pageins spike but
  per-process disk reads stay low, the churn is some other process, not us.
- **compressor + swap**: `vm.swapusage` (sysctl) + `vm_stat` "Compressor"/"Swapouts" pages —
  the E1 swap-collapse signature, for direct comparison.
- system available memory (`get_available_memory_gb`); total **ballast + server** footprint.

**Cache state (plan-review R1):** this is a **warm-cache** experiment (daily-driver realistic
— the model file is already resident from a prior request). sha256 verify + smoke test warm
the page cache; we then run a **standardized warmup** (one full decode request) before each
`-ngl` config's measurement window so all configs start from the same warm state. Cold-cache
is out of scope; `ri_diskio_bytesread` + pageins are logged so any unexpected cold misses are
visible in the record. Per-config warmup is identical → results are comparable.

### Required benchmark fields (project convention — plan-review R3)
Every `experiments.jsonl` record carries explicit fields (value or `null`+reason):
- `gpu_memory_mb` — proxy from llama.cpp startup **Metal model + KV + compute buffer sizes**
  (peak scratch isn't directly exposed; the banner sizes are the logged proxy, reason noted).
- `cache_hit_rate` — `null`, reason: "no GGUF page-residency API; see ri_diskio_bytesread /
  pageins as cache-miss proxy."
- `perplexity` — small fixed perplexity probe if cheap via the server, else `null`+reason
  with the 20-case harness as the quality gate.

## Method

### Phase A — Acquire model + smoke test (de-risk first)

**Pre-flight: calibrate `ri_diskio_bytesread` as a mmap-fault signal (plan-review R5-medium):**
Before using `ri_diskio_bytesread` as the authoritative churn gate, verify it actually captures
mmap page-fault reads on this macOS/llama-server setup. Run a calibration micro-benchmark
once (no ballast needed, just a fresh terminal):
1. Drop page cache for the GGUF file: `sudo purge` (or note residency level if purge is
   unavailable).
2. mmap-scan the first ~1 GB of the file cold, record `ri_diskio_bytesread` delta → expect a
   large read (the pages were faulted from SSD).
3. Re-scan the same range immediately (warm cache), record delta → expect near-zero reads
   (pages already in page cache, no disk faults).
4. Log both numbers. If the cold/warm delta is < 2× different, the counter is not capturing
   mmap faults reliably on this build — treat it as an informational metric only and note this
   in the experiment record; the pageout rate gate becomes the primary churn signal.
This step takes ~2 min and prevents a false-PASS from a non-functional churn gate.

1. Download Qwen3-30B-A3B **Q4_K_M GGUF** to a known path (HF: `unsloth/Qwen3-30B-A3B-GGUF`
   or `Qwen/Qwen3-30B-A3B-GGUF` — pick the one publishing Q4_K_M; record exact repo+file+sha).
   Use `huggingface-cli download` (single file). Verify file size + sha256.
2. Launch `llama-server` with the **pinned config** above (mmap on, `--jinja`, `--metrics`,
   `--port 8124`, `--ctx-size 8192`, etc.). Measure **two mandatory `-ngl` configs** — both
   are first-class, neither is conditional:
   - **`-ngl 999`** — full Metal offload, the normal Mac path. With Metal offload llama.cpp
     copies tensors into Metal buffers in unified memory, so mmap may only help load/source
     paging, not resident inference memory. Capture the startup **Metal model buffer size**.
   - **`-ngl 0`** — CPU-only, where mmap + page cache is genuinely exercised (weights stay
     mmap'd, OS pages them). Capture the startup **CPU model buffer size**.
   - **Partial-offload probe (mandatory if neither extreme passes; plan-review R3+R5):** the
     viable region is plausibly *between* the extremes — `-ngl 999` likely wires weights into
     Metal (MLX-like, fails footprint) while `-ngl 0` preserves mmap but may fail the ≥12
     tok/s gate. If neither extreme passes all gates, run a **bounded search** over a
     predefined candidate set: measure `-ngl` values at **25%, 50%, and 75% of the total
     layer count** (Qwen3-30B-A3B has ~94 transformer layers → ~24, ~47, ~71), then bisect
     once toward the crossover if the search reveals a clear boundary. Maximum 2 bisection
     steps total — do not extend into an open-ended tuning loop. All partial configs use the
     **same pinned ctx, ballast, mmap, and measurement protocol** and are fully
     **PASS-eligible**. Log the best-passing or closest-failing `-ngl` value.
   - **Verdict rule:** the gate verdict uses the best config that meets the **decode gate
     (≥ 12 tok/s)** while also satisfying TTFT + footprint + pageout + disk-churn + tool-call
     quality gates. A config that keeps footprint ≤ 8 GB but decodes < 12 tok/s does NOT pass.
     vm_stat pageins are logged diagnostically but are not a gate (system-wide proxy). If no
     config (999 / 0 / partial) passes all gates → honest FAIL.
3. Smoke test: one `/v1/chat/completions` tool call (reuse one harness case) returns 200
   with a parseable tool call. Confirms `--jinja` tool parsing works for Qwen3.
4. **Checkpoint** measured idle phys_footprint + startup buffer sizes per config to
   context.md before the long run.

### Phase B — 30-min sustained co-residency run (per viable config)

**Load order — two variants (plan-review R4-medium):** E1's failures were *load* failures, so
co-residency has two distinct questions. We test both and report both:
- **B1 — load-then-ballast (post-load survival):** start server, load model, *then* allocate
  ballast. Tests whether a running model survives an 8 GB working set appearing alongside it.
- **B2 — ballast-then-load (load-under-pressure):** allocate the 8 GB ballast *first*, then
  start the server and load the model. This directly mirrors the MLX failure story (model
  must load into an already-occupied machine). If B2 fails to load while B1 sustained fine,
  that itself is a publishable, sharper result.
The 30-min sustained-generation window + sampling below is run for **B1** as the primary gate;
**B2** is run at least as a load-success + short-sustain check (≥ 5 min) per viable `-ngl`
config — full 30-min B2 if time permits. Verdict notes which orderings passed.

1. Start `llama-server` (chosen config), wait for model load. (B2: ballast already held.)
2. (B1) In a separate driver process: allocate **exactly 8 GB fixed ballast** =
   `allocate_cpu_ballast(4.0)` + `allocate_mlx_ballast(4.0)` (NOT `create_memory_pressure`;
   see the ballast section above). Hold references for the full run. Log
   `get_available_memory_gb()` and ballast process phys_footprint before/after to confirm
   the 8 GB actually landed.
3. For **≥ 30 min sustained generation**: loop fixed decode requests (pinned prompt len,
   pinned `max_tokens`, non-streaming or SSE per existing adapter) back-to-back so the
   model is continuously generating, not idle.
4. **Sample every 30 s:** process-tree `phys_footprint` (median + max), `vm_stat` snapshot.
   Record per-request decode tok/s and first-token latency.
5. At end: compute sustained decode tok/s (median across run, exclude first warm request),
   p50 TTFT, pageout rate = `vm_stat_delta(pageouts)` / elapsed → MB/hr, phys_footprint
   median + max.

### Phase C — 20-case tool-call harness
1. With the server still up (post-ballast or under ballast — record which), run
   `scripts/h9_e1_harness.py --runtime served --port 8124`.
   - Verify model name the harness sends matches what llama-server expects (llama-server
     accepts any `model` string / the loaded alias) — adapt the harness `MODEL` constant
     or pass-through if needed.
2. Record success rate (semantic per-case grading) + structural vs recovered breakdown.

### Phase D — Verdict + logging
1. Log all metrics to `experiments.jsonl` via `log_experiment` using **phys_footprint**
   (full pinned config in `config`, gate results + pass/fail in `results`,
   `mem_method=phys_footprint`). One record per `-ngl` config measured (999 / 0 / partial),
   plus the harness record. Each record includes: phys_footprint median+max, **peak**
   phys_footprint, sustained decode tok/s, p50 TTFT, pageins+pageouts (MB and MB/hr),
   startup **CPU/Metal model buffer sizes** + **KV cache size** from the load banner,
   ballast confirmation (available before/after), GGUF repo/file/SHA + quant metadata,
   `cache_hit_rate=null` (reason: no GGUF page-residency API), `perplexity` (value or
   null+reason).
2. Write verdict to context.md using **explicit verdict classes (plan-review R2)**:
   - **FULL PASS** — all 6 gates met on B1 (30-min sustain) AND B2 load-success + ≥ 5-min
     sustain. E3 (#37) proceeds with llama.cpp mmap as the 30B daily-driver mode.
   - **CONDITIONAL PASS** — all 6 gates met on B1, but B2 fails to load or sustain. 30B is
     viable for post-load survival (e.g., restart-tolerant setups) but not guaranteed to load
     under a pre-occupied machine. Noted as a caveat; E3 may still proceed but must document
     the restriction.
   - **FAIL** — any gate fails on B1, or both B1 and B2 fail. Record definitively: concurrent
     tier = ≤ 4B only; 30B off-hours/batch only.

## Risks / failure modes (and how each is handled)

- **Metal offload negates mmap** (weights wired into unified memory like MLX): the central
  risk. Handled by measuring both `-ngl 999` and a CPU/partial config; the verdict reports
  which (if any) config keeps phys_footprint ≤ 8 GB while meeting the decode gate. If *no*
  config passes all four gates → honest FAIL.
- **Download fails / disk pressure**: ~18 GB file vs ~63 GB free. Verify free space before
  download; record exact repo/file. **Q4_K_M is mandatory for the #41 gate verdict**
  (plan-review R5): if only Q4_K_S is available, that run is **exploratory and NOT
  PASS-eligible** — record it as such; the verdict stays open until Q4_K_M is measured.
- **mmap still swaps under 8 GB ballast**: that is a legitimate FAIL measurement, not a bug.
  Record vm_stat evidence (matches E1's swap-collapse signature for direct comparison).
- **Harness model-name mismatch** with llama-server: adapt the served adapter's payload
  `model` field; llama-server is permissive. Smoke-test in Phase A before the 30-min run.
- **Long run interrupted**: sampling appends incrementally to a per-run log under
  `dev/active/issue-41-.../logs/` so a partial run is still analyzable; context.md
  checkpointed after each phase.

## Rollback / non-destructive notes

- No changes to existing scripts' behavior for the engine/vllm path; E1b adds a llama.cpp
  driver script (`scripts/h9_e1b_coresidency.py`) and reuses the harness/baseline helpers.
- The downloaded GGUF is large; record its path so it can be deleted after the experiment
  if disk is needed. Do not delete the MLX copy (used by E1/E2).

## Definition of done

Pass/fail verdict in this context file + `experiments.jsonl`; PR merged; issue closed.
PASS → #37 uses llama.cpp mmap 30B. FAIL → #37 re-scoped to 4B-only concurrent tier.
