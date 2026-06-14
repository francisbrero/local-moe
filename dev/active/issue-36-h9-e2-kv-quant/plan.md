# H9-E2 — KV quantization at workload context lengths

**Issue:** #36 | **Branch:** `experiment/h9-e2-kv-quant-workload-context`
**Depends on:** H7 (PR #20, kv4 validated at short ctx), E1 (#35 / PR #40), E1b (#41 / PR #42)

## Hypothesis

H7 validated `kv_bits=4, kv_group_size=64` with **zero perplexity loss**, but only on a
tiny model (24-layer, 896-hidden) at **short contexts**. The external claim is that KV
quantization overhead amortizes past ~4K tokens. The H9 daily workload that *drives* the
long-context requirement is **call-transcript summarization** (30–60 min calls ≈ 8–15K
tokens). E2 confirms H7's kv4 result holds at the **real workload context lengths
(10K / 12K / 15K)** on the actual Tier-1 model (Qwen3-30B-A3B) before E3/E5 adopt kv4 as
the default config, and quantifies the **effective context-per-GB** improvement.

## Scope — two arms (per the issue's 2026-06-12 rescope comment)

The E1 reframe split Tier 1 into **off-hours/batch** (MLX-wired, idle machine) and
**concurrent/daily-driver** (llama.cpp mmap, under ballast). E2 has one arm per mode.

### Arm (a) — Off-hours MLX tier (idle machine) — **primary, this PR**
MLX `kv_bits=4, kv_group_size=64` vs `kv_bits=16` baseline on Qwen3-30B-A3B-4bit, idle
machine, summarizing 10K / 12K / 15K-token call transcripts. This is the direct H7
re-validation at workload context lengths and the experiment the success gates are written
against. **Runs independently on an idle machine** — no E1b dependency.

### Arm (b) — Concurrent llama.cpp tier (under ballast) — **conditional, scoped but gated**
llama.cpp `-ctk q8_0` (and `-ctv q4_0` if stable on Metal) under 8 GB competing ballast,
mmap mode, where KV-cache pages compete with expert pages in the page cache. The issue
rescope says Arm (b) **should run after E1b (#41) confirms the mmap baseline passes
co-residency gates**. E1b's verdict (PR #42, recorded in
`research/h9-agentic-ops-stack.md`) was **CONDITIONAL / inconclusive on speed**: mmap
resident set ✓ (~4 GB) and tool-call quality ✓ (90%), but **decode under ballast was not
demonstrable** on the contended dev machine and **full Metal offload is GPU-OOM-blocked**.
Because the E1b concurrent-decode baseline never cleared its gate on this host, **Arm (b)
cannot produce a gate-eligible KV-quant result yet** — its blocker is the same host
contention, not KV quant. Decision below; default is to **defer Arm (b)** and ship Arm (a)
as the E2 deliverable, with the Arm-(b) driver wired but explicitly marked not-yet-eligible.

## What changes vs H7 (single-variable discipline)

| Axis | H7 (PR #20) | E2 Arm (a) |
|---|---|---|
| Model | tiny (24L / 896h, n_kv_heads=2) | **Qwen3-30B-A3B-4bit** (real Tier-1 model) |
| Context | short (≤ ~hundreds of tokens) | **10K / 12K / 15K** |
| KV configs | fp16 / kv8 / kv4 | **kv16 baseline vs kv4 (`group_size=64`)** |
| Quality metric | 4-domain perplexity | summarization spot-check + perplexity-on-transcript |
| Memory metric | RSS + peak_gpu_mb | **phys_footprint** (standing methodology) + MLX peak GPU |

The model, context, and quality task all change at once vs H7 — so E2 is **"does the kv4
verdict transfer to the real model at real context lengths,"** not a pure single-variable
swap. We log the model SHA + MLX version so the comparison is reproducible.

## Pinned configuration (gates are not reproducible without this)

All Arm (a) runs use these fixed parameters; every value is logged into the `config` block
of each `experiments.jsonl` record.

| Param | Value | Why |
|---|---|---|
| Model | `mlx-community/Qwen3-30B-A3B-4bit` (already cached) | Tier-1 model; same as E1 |
| Runtime | in-process `mlx_lm` 0.31.1 (`stream_generate`) | KV quant is an mlx_lm.generate kwarg; no server needed for a quality/memory probe. **Scope note (plan-review R5-R3-medium):** the issue says "vllm-mlx Tier 1 stack from E1," but the engine tier is the right place to isolate KV-quant quality+memory without HTTP/batching confounds (and vllm-mlx wires the same mlx_lm KV path). This experiment therefore **validates the mlx_lm engine tier**; the **verdict is capped accordingly** — "kv4 validated in-process; vllm-mlx served overhead (HTTP, continuous batching, server memory layout) is a follow-up before adopting kv4 as a *served* default." If time permits on the passing config, a short vllm-mlx served confirmation run is logged as a bonus (not a gate). |
| Context lengths | **10240, 12288, 15360** tokens (prompt) | The issue's 10K/12K/15K workload regime |
| KV configs | **A: `kv_bits=None` (fp16 baseline)**; **B: `kv_bits=4, kv_group_size=64`** | The H7 comparison, at workload ctx |
| `quantized_kv_start` | **0** | Quantize from the first token so the full prompt KV is quantized — this is the memory-relevant regime for a prefill-dominated summarization workload (a non-zero start would leave the bulk of a 15K prompt in fp16 and understate compression). H7 used the builtin default; we pin 0 explicitly and log it. |
| `max_tokens` (summary len) | **256** | Representative summary length; pinned for comparable prefill/decode split |
| Sampling | greedy (`temp=0`) | Deterministic; removes sampler variance from quality + tok/s |
| Samples per length | **5** call-transcript prompts | Issue: "spot-check on 5 samples per length" |
| Memory metric | **phys_footprint** (process tree, `proc_pid_rusage` RUSAGE_INFO_V2) + MLX `get_peak_memory` / `get_active_memory` | Standing H9 methodology: phys_footprint replaces RSS. MLX GPU counters isolate the **KV cache delta** (see below). |

## Transcript inputs (real-or-synthetic, reproducible)

The issue allows "real or synthetic meeting transcripts." Real call transcripts are PII and
not in-repo, so we **generate synthetic multi-speaker meeting transcripts** deterministically
(fixed seed via index, **no `Math.random`/`Date.now`** — varied by sample index) covering
realistic ops content (quarterly planning, customer escalation, eng standup, sales pipeline
review, hiring debrief). Each is token-trimmed to the exact target length using the model
tokenizer (`build_prompt_of_length` pattern from `h9_e1_baseline.py:284`). The summarization
instruction is a fixed prefix. We log the generator seed + the realized token count per
sample so inputs are reproducible. Transcripts are written under the experiment `logs/` dir
for inspection. **Limitation logged:** synthetic transcripts are coherent but not real call
data; the spot-check measures *relative* kv4-vs-kv16 degradation, which is robust to input
realism even if absolute quality is not call-representative.

## Measurement — isolating the KV cache (the core number)

The headline gate is **effective context per GB of KV cache**, so we must measure the **KV
cache memory specifically**, not just total footprint (which is dominated by the ~16.5 GB of
model weights and would swamp a KV delta of a few hundred MB).

**Method — analytic KV size is the PRIMARY gate metric; a steady-state GPU-counter reading
is the cross-check (plan-review R1-medium):** the naive `peak_active − baseline_active` delta
captures transient prefill buffers, graph/compiler allocations, and scratch — it does **not**
cleanly isolate the KV cache and would *inflate* it. So we invert the original framing:

1. Load model. `mx.eval()` to force materialization; record baseline `mx.get_active_memory()`
   (weights + framework, no cache).
2. **Direct cache `.nbytes` = the gate metric (verified API, plan-review R5-R2).** After
   prefill, each cache entry is a `KVCache` (kv16) or, with `quantized_kv_start=0`, a
   **`QuantizedKVCache`** (verified: `maybe_quantize_kv_cache` calls `c.to_quantized(...)` once
   `c.offset >= quantized_kv_start`). Both expose a **`.nbytes` property** — so the KV-cache
   size is read **directly and exactly** as `sum(c.nbytes for c in prompt_cache)`, not inferred
   from allocator deltas. This is the gate metric. We **cross-check** against the analytic
   formula `kv16 bytes = 2 (K+V) × n_layers × n_kv_heads × head_dim × seq_len × 2 (fp16)` and
   kv4 ≈ kv16/3.5 (group_size=64 + per-group scale/bias; matches H7's 3.56×). Model dims come
   from the loaded config (`n_layers`, `n_kv_heads`, `head_dim`), logged per record.
   **Assertion (plan-review R5-R2-medium):** in the kv4 runs, assert every cache entry
   `isinstance(c, QuantizedKVCache)` after prefill — if any entry is unquantized, the kv4 path
   did not engage → `status=infra_fail_kv_quant_unsupported` (not a quality result).
3. **Steady-state GPU-counter cross-check (not the gate):** build a `prompt_cache` via
   `make_prompt_cache(model)`, run `stream_generate` with the pinned KV config over the full
   prompt + `max_tokens`, then **`mx.eval()` and read `get_active_memory()` while the
   populated cache is still retained** — `cache_resident_active − baseline_active` is the
   steady-state cache footprint (measured *after* prefill scratch is freed, not a peak delta).
   `mx.get_peak_memory()` is recorded **separately, only for total-budget / OOM monitoring**.
4. **`.nbytes`-vs-analytic agreement is a HARD VALIDITY GATE (plan-review R2-R2-medium):**
   validate agreement on the **Phase A controlled run first**. The direct `.nbytes` figure is
   the per-GB gate metric; the analytic formula is the cross-check. If they diverge > 25%
   (unexpected — both are exact-ish), the cell is **`status=memory_inconclusive`** and we
   investigate before trusting the ratio. The GPU-counter steady-state delta (point 3) is an
   additional sanity check on total allocation but is **not** the per-GB number now that
   `.nbytes` gives the cache size directly. A cell counts as a clean per-GB PASS when `.nbytes`
   and analytic agree within 25% AND the ratio ≥ 1.5×.
5. **phys_footprint** of the process tree is logged at baseline and peak for the standing
   methodology and as the total-memory budget check — but the **per-GB gate uses the analytic
   KV-cache figure** (cross-checked by the steady-state delta), since that is what kv4 shrinks.

**Effective context per GB** = `context_tokens / kv_cache_GB`, computed per (length, config).
Gate: `(tokens/GB)_kv4 / (tokens/GB)_kv16 ≥ 1.5×` at each length. (H7's 3.56× compression
predicts ~3.5× here; the 1.5× gate has comfortable headroom, and a result far below 3.5×
would itself be a finding about large-model KV layout.)

## Quality measurement (no measurable degradation gate)

Per the issue ("ROUGE or manual spot-check on 5 samples per length"):
1. **Teacher-forced PPL on the SUMMARY-TOKEN POSITIONS ONLY (primary, automatable) — same
   tokens under both configs (plan-review R3-R1-medium):** PPL is only comparable if both
   configs score the **identical** token sequence, and it must measure the quantity that
   actually matters — the summary distribution, not the prompt. So: pin **one target per
   prompt = the kv16-generated summary** (generated first; deterministic under greedy). Run
   the full sequence (transcript prompt + summary) teacher-forced under each KV config, but
   **compute NLL/PPL over the summary-token positions only** (conditioned on the full
   transcript via the KV cache). A 10–15K prompt would otherwise swamp a 256-token summary and
   hide any degradation — so the gate uses **summary-only ΔPPL**. The full-sequence /
   transcript PPL is kept as a **diagnostic** but is not the gate. Scoring target text + its
   token count logged per record. Per-length mean **summary-token ΔPPL (kv4 − kv16)**; gate:
   **|ΔPPL| ≤ 1% relative** (matching H7's "zero loss").

   **Scoring MUST route through the quantized cache (plan-review R5-R2-medium) — else the gate
   proves nothing.** A naive single full-sequence forward pass builds a *fresh* default
   (fp16) cache and would report identical PPL for both configs, silently not testing kv4. So
   the scoring run **reuses the generation cache path**: prefill the transcript into a
   `prompt_cache`, apply `maybe_quantize_kv_cache(..., kv_bits=4, kv_group_size=64,
   quantized_kv_start=0)` so the cache becomes `QuantizedKVCache`, then **teacher-force the
   pinned summary tokens one step at a time through `model(token, cache=prompt_cache)`**,
   accumulating NLL over the summary positions — the same incremental path `generate_step`
   uses, so the quantized K/V actually participate. **Assert** `isinstance(c, QuantizedKVCache)`
   for every entry before scoring the kv4 variant; if not, the PPL gate is
   **infrastructure-invalid** (`status=infra_fail_kv_quant_unsupported`) and we fall back to
   the generated-summary ROUGE-L + manual spot-check as the quality signal, logged as such.
2. **ROUGE-L between kv4 and kv16 summaries (secondary):** with greedy decoding, kv16 and
   kv4 should produce near-identical summaries if quantization is lossless. Compute ROUGE-L
   of kv4-summary vs kv16-summary (kv16 as reference) per sample. ROUGE-L ≈ 1.0 = no
   degradation. `rouge-score` is a tiny pure-Python dep; if unavailable we fall back to
   token-level exact-match ratio + normalized edit distance and log the method.
3. **Manual spot-check artifact:** the 5×3 = 15 summary pairs (kv16, kv4) are written to
   `logs/summaries/` for human inspection; context.md records a one-line spot-check verdict
   per length.

**Gate composition (plan-review R6-low — ROUGE-L is brittle for generated text):** the
**formal quality gate is teacher-forced ΔPPL ≤ 1% relative** at each length (the rigorous,
metric-stable signal), **plus the manual spot-check showing no degradation**. ROUGE-L is a
**divergence trigger, not a hard gate**: if mean ROUGE-L(kv4 vs kv16) drops **below 0.90** at
any length, that flags the spot-check for closer human review at that length (small lexical
differences under greedy can lower ROUGE without real quality loss). The quality gate
**PASSES** when ΔPPL is within noise at all three lengths and the spot-check finds no
degradation, with no OOM at 15K; a low ROUGE-L escalates review but does not by itself fail
the gate.

## Success Gates (from the issue)

| Gate | Target | Source metric |
|---|---|---|
| Summarization quality holds at 10K/12K/15K | no measurable degradation vs kv16 | **ΔPPL ≤ 1% rel (formal) + spot-check**, per length; ROUGE-L < 0.90 triggers extra review but is not itself a fail (plan-review R6) |
| Effective context per GB | **≥ 1.5×** vs kv16 | analytic KV-cache GB is the gate metric **only while it agrees with the measured steady-state delta within 25%** (hard validity gate); on > 25% divergence the cell is `memory_inconclusive` and the gate falls back to the **measured** delta (plan-review R2-R2-medium) |
| No OOM / degradation at 15K | kv4 completes 15K without OOM | run completes; phys_footprint logged |
| Speed amortization (reporting, non-blocking) | kv4 prefill/decode not materially slower than kv16 | median prefill tok/s, decode tok/s, TTFT per (L,cfg); **flag if kv4 is >10% slower** than kv16 at any length (plan-review R2 — the amortization hypothesis is about overhead; flagged, not a hard fail) |

## Method

### Phase A — De-risk (smoke test, idle machine)
1. Confirm idle machine (quit Cursor/Chrome/extra Claude procs per the
   [machine-contention blocker](../../../memory note)); log `get_available_memory_gb()`.
   MLX-wired 30B is ~16.5 GB — it loads fine **only** on a quiesced machine (E1 lesson).
2. **Per-length analytic preflight — explicit budget basis, no double-counting
   (plan-review R2-medium, R5-R1-medium):** the comparison must not double-count weights.
   Two clearly-separated checks:
   - **Pre-load (total-budget) check:** measured **once, before loading the model**, against a
     fixed usable cap. `total_required(L,cfg) = weights (~16.55) + analytic_KV(L,cfg) +
     scratch (~2) + safety (~1)` vs **usable cap ~19–21 GB** (idle 24 GB host). This is the
     "could the whole thing fit at all" gate.
   - **Post-load (incremental-headroom) check:** measured **after the model is loaded**, where
     weights are already resident. Compare only the **incremental** need
     `analytic_KV(L,cfg) + scratch + safety` against **post-load** `get_available_memory_gb()`
     (which already reflects the resident weights). This avoids charging the 16.55 GB twice.
   A run proceeds only if **both** checks pass; otherwise skip, log
   `status=skipped_oom_preflight` with **both total footprint and incremental headroom**
   logged. The 15K **kv16** baseline is the highest-risk cell (largest fp16 KV); it gets its
   own gate (see below).
3. Load `mlx-community/Qwen3-30B-A3B-4bit`; record baseline active GPU mem + phys_footprint.
4. **kv4 compatibility gate (plan-review R3-medium) — verify quantization is real, not a
   silent no-op:** run a short and a medium prompt under kv16 and kv4, and confirm kv4
   **actually changes the cache representation and memory slope** — i.e. the steady-state
   cache delta grows ~3.5× more slowly with seq_len under kv4 than kv16 (a measurable slope
   difference between the short and medium points), and the cache objects are
   `QuantizedKVCache`. If kv4 does **not** change the memory slope, MLX's quantized-KV path is
   unsupported/broken for this model+version → classify as **INFRASTRUCTURE FAILURE**
   (`status=infra_fail_kv_quant_unsupported`), not a model-quality failure, and stop. This
   distinguishes "kv4 doesn't help" from "kv4 isn't actually engaged."
5. **Empirical per-length preflight (plan-review R1-medium) — analytic budget is planning
   input, not sufficient proof:** for each length L, before the 5-sample sweep, build the
   **prompt cache only** (prefill, no long decode) under the run's config, `mx.eval()`, and
   record peak MLX memory + phys_footprint. Proceed to the full sweep for (L, config) **only
   if observed headroom ≥ ~1 GB** vs `get_available_memory_gb()`; otherwise log
   `status=skipped_oom_empirical` with the observed peak. This catches allocator
   fragmentation, tokenizer/input buffers, attention workspaces, and dual-cache overlap that
   the analytic estimate omits — the kv16 12K/15K cells are the tight ones on this host.
   **Log the per-cell expected budget breakdown before each launch** (plan-review R4-low):
   `weights ~16.55 + analytic_KV(L,cfg) + scratch ~2 + safety ~1` → expected total. Reference
   figures: analytic kv16 KV ≈ 1.41 GiB at 15K (~0.40 GiB kv4) → kv16 15K ≈ 20.96 GB total,
   kv4 15K ≈ 19.95 GB — both borderline on a 24 GB host, viable only genuinely idle. The
   logged budget makes a skip self-explaining.
6. Checkpoint to context.md before the full sweep.

**kv16-OOM reporting path — verdict split by gate (plan-review R2-medium, R3-R3-medium):** if
the **15K kv16** baseline OOMs/fails preflight while **15K kv4 completes**, that is **not** an
E2 failure — it is a *positive* demonstration of kv4's value (kv4 fits where kv16 cannot). But
the two gates at 15K must be reported **separately**, because a missing kv16 baseline means
there is nothing to compare quality against:
- **Memory / per-GB gate at 15K = PASS-with-asterisk:** per-GB computed against the
  **analytic** kv16 size (well-defined even if kv16 can't physically run); verdict states
  "kv16 15K not physically runnable on this 24 GB host; kv4 15K runs — strongest form of the
  per-GB result."
- **Quality gate at 15K = `unverified`** — without a kv16 15K summary there is no teacher-
  forced ΔPPL or ROUGE comparison; we do **not** claim "no degradation vs kv16" at 15K.
  Quality is proven only at the lengths where both configs ran (10K/12K).
The **overall verdict** then reads explicitly: "quality proven through 12K; 15K quality
unverified (kv16 OOM); kv4 memory advantage demonstrated at all three lengths." Logged
distinctly from a clean both-run PASS.

### Phase B — KV sweep (the experiment)
For each length L ∈ {10240, 12288, 15360} and each config C ∈ {kv16, kv4}:
1. Generate the 5 synthetic transcript prompts for L (fixed per index), tokenizer-trimmed.
2. For each sample: fresh `prompt_cache`; `reset_peak_memory()`; `stream_generate` with C;
   record prefill tok/s, decode tok/s, TTFT, generated summary, KV-cache GPU delta, peak
   phys_footprint. Compute perplexity-on-transcript under C.
3. Aggregate per (L, C): median prefill tok/s, median decode tok/s, mean PPL, KV GB.
4. Write summaries to `logs/summaries/`.

### Phase C — Analysis + gates
1. Per length: ΔPPL (kv4−kv16), ROUGE-L(kv4 vs kv16), effective-context-per-GB ratio.
2. Evaluate the three gates. Record PASS/FAIL per gate per length.

### Phase D — Verdict + logging
1. Log one `experiments.jsonl` record **per (length, config)** via `log_experiment`
   (`mem_method=phys_footprint`), full pinned config in `config`, gate results in `results`,
   plus a summary record with the per-GB ratios and overall verdict. Required project fields
   per record (value or `null`+explicit reason string, plan-review R4-low):
   - `gpu_memory_mb` — MLX peak active memory (`get_peak_memory`) for the run.
   - `cache_hit_rate` — `null`, `cache_hit_rate_reason: "not applicable: no expert/page cache
     in the MLX KV-quant arm"`.
   - `perplexity` — the pinned-target PPL value under the run's KV config.
   - `kv_cache_gb_analytic` (gate metric) + `kv_cache_gb_measured` (steady-state cross-check).
2. **Verdict is scoped to the 24 GB off-hours MLX tier (plan-review R1-medium):** this Arm (a)
   validates MLX-wired Qwen3-30B-A3B on an **idle 24 GB M4 Pro** (~19–21 GB usable). It does
   **not** clear the original CLAUDE.md 16 GB / ~10–11 GB-usable daily-driver budget — at
   ~16.5 GB weights the model does not even fit that budget. The verdict must state "validated
   for the 24 GB off-hours MLX tier; the 16 GB daily-driver budget is the concurrent-tier
   question (mmap/offload/streaming — E1b/E3/E5), not addressed here." kv4 *helps* a 16 GB
   target but is not *sufficient* for it; that is not in scope.
3. Verdict classes:
   - **PASS** — quality holds at all 3 lengths AND per-GB ≥ 1.5× at all 3 lengths AND no OOM
     at 15K. → E3/E5 adopt `kv_bits=4, kv_group_size=64` as default for the **24 GB off-hours
     tier**.
   - **PARTIAL** — passes at 10K/12K but degrades or OOMs at 15K. → kv4 default capped at the
     passing context; note the ceiling for E3.
   - **FAIL** — measurable degradation or < 1.5× per-GB at workload lengths. → kv4 not adopted
     as default; record the negative result (valuable — contradicts the H7-extrapolation).

### Arm (b) decision (recorded, not silently skipped)
Arm (b)'s prerequisite — a clean E1b concurrent-decode baseline — **did not clear its gate on
this host** (E1b verdict: decode-under-ballast not demonstrable; GPU-OOM on full offload).
Therefore Arm (b) **cannot yield a gate-eligible KV-quant number** until the host can be
genuinely quiesced for a partial-offload baseline. **Default plan: defer Arm (b)**; ship
Arm (a) as E2's deliverable; leave the llama.cpp `-ctk q8_0 / -ctv q4_0` driver wired but
labeled *exploratory, not PASS-eligible* (mirrors E1b's Q4_K_S handling). If the user wants
Arm (b) attempted opportunistically on a quiesced machine, the driver supports it; any
numbers it produces are logged with `gate_eligible=false` and the same host-contention caveat
E1b carried. This keeps E2's verdict honest and unblocked rather than waiting on the #41 host
blocker.

## Risks / failure modes

- **30B won't load co-resident** (E1's failure): Arm (a) is an **idle-machine** experiment by
  design; we gate on `get_available_memory_gb()` ≥ ~18 GB before loading and abort with a
  clear message otherwise (do not thrash). The machine-contention blocker memory applies.
- **KV delta is too small to measure cleanly** against 16.5 GB of weights: handled by using
  MLX GPU counters (which isolate allocations) + the analytic cross-check; the analytic
  figure is the documented fallback gate metric.
- **`quantized_kv_start` semantics:** pinned to 0 and logged; if mlx_lm requires a non-zero
  warmup before quantizing, we record the realized quantized-fraction and adjust the analytic
  baseline accordingly (still report the honest measured compression).
- **ROUGE not deterministic across configs even when lossless** (greedy should make them
  near-identical): if kv4 and kv16 greedy summaries diverge token-for-token, that **is** a
  degradation signal at that length, not a metric artifact — reported as such.
- **15K exceeds model trained context / KV OOM:** Qwen3-30B-A3B supports ≥ 32K context, so
  15K is in-range; if KV OOM occurs under kv16 it is itself the finding (kv4 is what avoids
  it) — logged, not hidden.

## Rollback / non-destructive notes

- E2 adds one driver script (`scripts/h9_e2_kv_workload.py`) and reuses
  `h9_e1_baseline.py` (phys_footprint collector, `build_prompt_of_length`,
  `engine_generate_timed`) and `experiment_utils.py` (`log_experiment`,
  `get_available_memory_gb`, `get_environment_info`). No behavior change to existing scripts.
- Uses the already-cached MLX 4-bit model; downloads nothing. Synthetic transcripts and
  summaries are written under the experiment `logs/` dir only.

## Definition of done

**Scope of this PR = Arm (a) only (plan-review R5-low).** Arm (b) is explicitly a
**follow-up / out-of-scope** here (blocked on the E1b host-contention baseline); its driver
may be wired but its numbers are not gate-eligible and do not gate this PR.

Done when: pass/fail verdict **per length for Arm (a)** in this context file +
`experiments.jsonl`; PR merged; issue closed. PASS → E3/E5 default to kv4 (off-hours tier).
Arm-(b) status recorded as a follow-up with its host-contention caveat (does not block DoD).
