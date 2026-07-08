# Issue #44 — H9-E6: Detached quiesced measurement run

## Goal

One bundled run script `scripts/h9_e6_detached_run.py` that, on a quiesced machine
(Cursor quit, ≥19 GB free), produces at least one recorded number in `experiments.jsonl`
for each of E1b (partial-offload decode), E2 (KV-quant sweep, both arms), and E4 (idle MLX
baseline). The whole point of the prior E1/E2/E1b failures was **machine contention from the
orchestrating IDE**; this script is the detached, hard-gated remedy.

## Key constraints (from prior experiments — must preserve)

- **Metric convention:** `phys_footprint` (via `proc_pid_rusage RUSAGE_INFO_V2`) is authoritative
  for MLX-wired memory; `ri_resident_size` is the working-set metric for mmap/llama.cpp.
  RSS is never a gate metric.
- **GPU-OOM finding:** full Metal offload (`-ngl 999`) of 30B is not viable (18186 MiB GPU budget).
  E1b arm must use **partial offload** with prompt cache trimmed (`--cache-ram` low).
- **Detached-only:** preflight aborts non-zero if free RAM < 19 GB. This is by design; running
  from inside Cursor will (correctly) abort.

## Reuse map (do NOT re-implement)

| Need | Source |
|---|---|
| experiments.jsonl logging | `experiment_utils.log_experiment(name, phase, config, results, status, env)` |
| env info | `experiment_utils.get_environment_info()` |
| available RAM | `experiment_utils.get_available_memory_gb()` |
| rusage struct/libc | `h9_e1_baseline`: `_RUsageInfoV2`, `_RUSAGE_INFO_V2`, `_libc`, `tree_pids`, `phys_footprint_bytes`, `tree_footprint_gb` |
| tree counters (phys/resident/pageins/diskread) | `h9_e1b_coresidency.tree_counters(root_pid)` |
| served timing (ttft, decode tok/s) | `h9_e1_baseline.served_generate_timed` |
| E1b primitives | `h9_e1b_coresidency`: `wait_healthy`, `Ballast`, `tree_counters`, `swap_used_gb`, buffer-size parse |
| (reference only — NOT reused) | `h9_e1b_coresidency.run` / `_compute_gates` — teardown + wrong mmap metric; see Phase A override |
| E2 MLX kv4 arm + preflight | `h9_e2_kv_workload`: `cmd_run`, `preflight`, `_summarize_once`, `teacher_forced_nll` |
| E4 MLX baseline (decode/prefill/phys) | `h9_e1_baseline.run_engine`, `_assemble` |
| 20-case tool-call harness | `h9_e1_harness` (`--runtime served` for E1b llama-server, `--runtime engine` for E4 MLX) |
| ballast (fixed 8 GB) | `h9_e1b_coresidency.Ballast` |
| vm_stat pageout delta | `experiment_utils.get_vm_stat`, `vm_stat_delta` |

## The one thing that does NOT exist yet

**E2 Arm 2 (llama.cpp `-ctk q8_0` KV quant)** has no code. Must be written: launch
`llama-server` with `-ctk q8_0` (+ `-fa` flash-attn required for KV quant) under an 8 GB ballast,
run a summarization workload, measure ΔPPL-ish quality proxy + pageout rate vs a no-KV-quant
mmap baseline. Model to compare against: `-ctk f16` (default) baseline arm.

## Decisions (confirmed with user)

- **Scope:** build the full script + self-test the preflight, then attempt the run now
  (expected to abort at preflight from this Cursor session — that abort IS a valid, documented outcome).
- **E1b `-ngl`:** parameterize with `--e1b-ngl` (default `24` = 50% of 48 layers), `--cache-ram`
  trimmed (default `512` MiB) to dodge the GPU-OOM. Sweepable via repeated CLI runs.

## Phased design of `scripts/h9_e6_detached_run.py`

> **Execution order:** Phase 0 (preflight) → Phase C (E4 idle) → Phase A (E1b) → Phase B (E2) →
> Phase D (summary). Section headers below are labelled by experiment (A=E1b, B=E2, C=E4); the
> letters are historical, the ordering above is authoritative.

### Phase 0 — Preflight (hard gate)
- Print a `HOW TO RUN DETACHED` banner (manual procedure comment block, also echoed at runtime).
- Measure available RAM via `get_available_memory_gb()`. If `< 19.0 GB`: print human-readable
  shortfall (how much free, how much needed, "quit Cursor & Chrome, rerun from Terminal.app"),
  log an `aborted` record to experiments.jsonl for auditability, `sys.exit(2)`.
- `--force` flag to bypass ONLY for dry-run/self-test (still logs, still warns loudly).

### Phase A — E1b partial-offload decode
- **Do NOT delegate to `h9_e1b_coresidency.run()`** — it kills `llama-server` in its own teardown
  before we could run the served tool-call harness, its `start_server()` doesn't accept
  `--cache-ram`, and `_compute_gates()` gates mmap footprint on `phys_footprint` (wrong metric).
  Instead orchestrate at the lower-level primitives inside a `with` / `try...finally`:
  1. Stand up 8 GB `Ballast` (E1b `Ballast` class).
  2. Launch `llama-server` ourselves at `--e1b-ngl` (default 24) with `--cache-ram 512`, `--jinja`,
     `--metrics`, mmap default. The E6 local `start_server` **copies** E1b's arg list (it does NOT
     call `h9_e1b_coresidency.start_server`, whose signature has no `cache_ram` param) and appends
     `["--cache-ram", str(cache_ram_mb)]` before `Popen`. Parse startup buffer sizes from server stderr (Metal model buffer, KV
     cache size/type, prompt cache) into logged fields.
  3. Run the sustained sampling loop (30 min, `--e1b-minutes`) reusing E1b's `tree_counters` +
     `served_generate_timed` + `Ballast.retouch()`: decode tok/s median, ttft p50, pageout rate
     (vm_stat delta), `ri_resident_size` p95/max, diskio churn.
  4. **While the same server is still alive**, run the tool-call harness
     (`h9_e1_harness --runtime served --port 8124`) and capture semantic_pass_rate.
  5. `finally`: kill server, drop ballast, `gc.collect()`.
- Compute gates ourselves (do not reuse `_compute_gates` verbatim): mmap footprint gate on
  **`ri_resident_size` p95/max ≤ 8 GB** (NOT phys_footprint), decode ≥12, pageouts <200 MB/hr,
  tool ≥90%.
- **Full unified-memory budget gate (NEW):** `ri_resident_size` alone can pass while total pressure
  thrashes. After server load, compute and log a budget line:
  `ballast(8) + Metal_model_buffer(parsed) + KV_cache(parsed) + prompt_cache(parsed) +
  mmap_working_set(ri_resident_size) + 1 GB headroom` must fit within usable memory (~20 GB).
  If it exceeds, log a `budget_exceeded` warning in the record (do not silently pass); document
  the safe `--e1b-ngl` range (higher ngl = larger Metal buffer). Log every component separately.
- Log `h9_e6_e1b` record.

### Inter-phase memory hygiene (applies between EVERY phase)
Each phase that stands up a server or ballast runs inside a `try...finally` that guarantees
teardown. After teardown, a shared `cooldown_and_gate(min_free_gb)` helper:
`gc.collect()` → `mx.clear_cache()` (gated on an explicit module-level `mlx_touched` bool set at
the start of any MLX phase; behind a try/import guard) → short sleep (allow compressor/page-cache
to settle) → re-read `get_available_memory_gb()` + record swap/pageout state. If free RAM is below
the phase's requirement it **logs a `skipped_lowmem` record for the next phase and skips it**
(rather than OOM/thrash). MLX phases require ≥19 GB free; llama.cpp+ballast phases require the
8 GB ballast to have been released and free RAM back above 19 GB before the next MLX phase.

### Phase ordering (E4 runs FIRST to protect the "idle baseline" claim)
Ordering is **E4 → E1b → E2-MLX → E2-llamacpp**. E4 is the idle MLX baseline; running it after a
30-min llama.cpp run + E2 sweep would carry over page-cache/swap/allocator state and taint the
"idle" claim. So E4 runs immediately after preflight while the machine is genuinely idle. Each
subsequent phase is guarded by `cooldown_and_gate`. Rationale recorded in the E4 result record
(`ran_first: true`, idle-state snapshot).

### Phase C (runs FIRST) — E4 idle MLX baseline
- Reuse `h9_e1_baseline.run_engine` path: decode tok/s @8K, prefill tok/s @8K & 16K,
  phys_footprint peak. Run tool-call harness `--runtime engine` (20 cases, MLX).
- Log `h9_e6_e4`. Gate: decode ≥20, prefill ≥300 @8K, tool ≥90%, phys ≤21 GB.

### Phase A (runs after E4) — E1b partial-offload decode
(see Phase A block above — orchestrated at low-level primitives, mmap gate on ri_resident_size.)

### Phase B (runs last) — E2 KV sweep
- **Arm 1 (MLX kv4):** reuse E2 `preflight` + `_summarize_once` + `teacher_forced_nll` at
  kv_bits∈{16,4}, lengths from issue (~10–15K), measure paired ΔNLL quality + effective-context/GB
  vs FP16. Log `h9_e6_e2_mlx`. Gate: ΔNLL within E2 harness threshold AND ≥1.5x ctx/GB vs FP16.
- **Arm 2 (llama.cpp q8_0):** NEW. Concrete quality method (was underspecified):
  - Launch `llama-server -fa -ctk q8_0` (PORT **8126**) and separately a `-ctk f16` baseline
    (PORT **8125**), each under 8 GB ballast — distinct ports from E1b's 8124 so a not-yet-released
    port from Phase A can't block bind.
  - **Corpus (self-generated, no dependency on any prior E2 run):** the transcripts dir is empty
    and no prior E2 run populated it. The E6 script generates its own deterministic corpus on first
    run using E2's `_make_transcript`/`build_transcript_prompt` with pinned `sample_idx` values and
    lengths, saves to `dev/active/issue-44-h9-e6-detached-run/corpus/`, and logs each file's path +
    SHA256 in the record. Both arms score the identical corpus.
  - **Quality scorer (capability-probed, no unavailable binary):** `llama-perplexity` does NOT
    exist in the install (only `llama-server`), so it is NOT a fallback. **Known limitation:**
    llama-server's `/completion` `n_probs` returns top-k logprobs for *sampled* tokens only, NOT
    for prompt/input tokens — so teacher-forced NLL over a fixed target is generally not
    obtainable via this REST API. The plan therefore expects the quality sub-gate to degrade to
    `None`; it must not fabricate a number. Method:
    1. Probe once at startup: POST `/completion` with `n_probs>0` on a short prompt, and
       **verify the returned per-token logprob array length equals the prompt token count** (i.e.
       input-token coverage), not merely that a logprobs field is present. Only input-token
       coverage counts as "supported". Log `quality_method`.
    2. If (unexpectedly) supported: teacher-force-score the fixed corpus targets under both
       `-ctk f16` and `-ctk q8_0` → mean NLL per arm. Pass: relative ΔNLL(q8_0 vs f16) ≤ 1%.
    3. If NOT supported (the expected case): record `quality_method="unavailable"` +
       reason "llama-server REST exposes sampled-token logprobs only, not input-token", set the
       quality sub-gate `None`/na_reason, and rely on the pageout/memory numbers (which need no
       logprobs). The run still produces numbers for this arm, satisfying the "≥1 number" gate.
  - **Startup observability (logged fields):** parse server stderr for Metal model buffer, CPU
    model buffer, KV cache size + type (`q8_0`/`f16`), prompt cache size, `-ngl`, `--cache-ram`,
    flash-attn flag. Use pageins + `ri_diskio_bytesread` delta as the page-cache/cache-hit proxy.
  - **Pageout gate (absolute AND relative):** q8_0 pageout rate < 200 MB/hr (absolute) AND
    q8_0 ≤ f16 + 200 MB/hr (relative). A thrashing f16 baseline must not let q8_0 pass on the
    relative gate alone. Log vm_stat deltas for both arms.
  - Log `h9_e6_e2_llamacpp` (both arms + delta).

### Phase D — Overall verdict
- Emit `h9_e6_SUMMARY` record aggregating pass/fail per sub-experiment + whether every arm
  produced ≥1 number (the overall run gate).

### Launch procedure
- Manual `HOW TO RUN DETACHED` comment block at top of script.
- Optional: `scripts/h9_e6.plist` launchd template for 2 AM overnight run (documented, not installed).

## Testing strategy (from this session)

- Unit-verify preflight math (abort < 19 GB) with `--force`-off dry path.
- Import-check every reused symbol resolves (no runtime AttributeError).
- Lint: `ruff check scripts/h9_e6_detached_run.py`.
- Attempt the real run per user request → expect preflight abort (documented outcome), OR
  if machine is quiesced, real numbers.

## Post-run documentation (always, incl. after a preflight-abort self-test)
After the detached run (or the abort self-test from this session), update
`context.md` (results / abort reason, review-round counts) and `tasks.md` (completion state).

## Out of scope
- Actually merging PR #43 / editing #37, #38 (those are follow-ups on completion, done after
  real numbers exist).
- Producing the overnight numbers from *this* session if preflight aborts (that's the user's
  detached run).
