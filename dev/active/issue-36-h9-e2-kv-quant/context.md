# H9-E2 — Context / Findings

**Issue:** #36 | **Branch:** `experiment/h9-e2-kv-quant-workload-context`

## Current state
- Phase 0 setup complete: branch created, docs scaffolded.
- Plan drafted (`plan.md`), under automated plan-review loop.

## Environment (verified)
- MLX 4-bit model cached: `mlx-community/Qwen3-30B-A3B-4bit` (`~/.cache/huggingface`).
- GGUF cached: `models/gguf/Qwen3-30B-A3B-Q4_K_M.gguf` (19 GB) — for Arm (b) only.
- `mlx_lm` 0.31.1; `generate_step`/`stream_generate` accept `kv_bits`, `kv_group_size`,
  `quantized_kv_start` (see `mlx_lm/generate.py:303`, `maybe_quantize_kv_cache:295`).
- MLX GPU counters available: `mx.get_active_memory`, `mx.get_peak_memory`,
  `mx.reset_peak_memory` — used to isolate the KV-cache delta.
- Reusable: `h9_e1_baseline.py` (phys_footprint RUSAGE_INFO_V2 collector,
  `build_prompt_of_length`, `engine_generate_timed`), `experiment_utils.log_experiment`.

## Baselines being compared against
- H7 (PR #20): kv4/group64 = zero PPL loss, 3.56× compression — but tiny model, short ctx.
- E1 (#35): MLX-wired 30B = 16.55 GB phys_footprint; loads only on a quiesced machine.
- E1b (#42): mmap resident ~4 GB, tool-call 90%; concurrent decode not demonstrable on the
  contended host → Arm (b) prerequisite unmet.

## Implementation
- `scripts/h9_e2_kv_workload.py` implemented (`smoke` + `run` subcommands).
  - Deterministic synthetic multi-speaker transcripts (seeded by sample index, no RNG/clock).
  - **Direct KV-cache sizing via `sum(c.nbytes)`** (gate metric) + analytic cross-check
    (hard 25% validity gate). Verified analytic ratio = **3.56×** (matches H7 exactly).
  - kv4 routes through `make_prompt_cache` + `maybe_quantize_kv_cache(quantized_kv_start=0)`;
    asserts every entry is `QuantizedKVCache`. Teacher-forced summary-only ΔPPL through the
    same incremental cached path (so quantized K/V actually participate).
  - Two-stage memory preflight (pre-load total vs post-load incremental; no weight
    double-count). kv16-OOM-at-15K path: memory PASS-with-asterisk, quality = unverified.
  - ruff clean; analytic/rouge unit-checks pass.

## Findings
- **Machine-contention blocker hit (expected):** free RAM at implementation time = **10.9 GB**,
  below the ~18 GB the MLX-wired 30B (16.55 GB) needs. The sweep is an **idle-machine**
  experiment by design — the script's preflight aborts rather than thrash (E1 lesson). The
  measurement runs (`smoke`, then `run`) must be executed on a **quiesced machine** (quit
  Cursor/Chrome/extra Claude procs) per [h9-machine-contention-blocker]. No measured E2
  numbers yet; code + gates are ready to run.
- _(Phase B/C results to fill in after the quiesced sweep.)_

## Review stats
- **Plan review rounds: 6** — 12 findings addressed; converged round 6 (reviewer confirmed all
  remaining items already covered). Key hardening: summary-only teacher-forced ΔPPL (not
  prompt-swamped); `.nbytes` as the exact per-GB gate metric with a hard 25% analytic-validity
  gate; two-stage memory preflight (no weight double-count); kv16-OOM-at-15K verdict split
  (memory PASS-with-asterisk, quality unverified); verdict scoped to the 24 GB off-hours tier.
- **Code review rounds: 3** — converged round 3 (no material findings). Findings addressed:
  - R1 (4 medium): chunked prefill (512) with interleaved quantization to match mlx_lm
    `generate_step` production semantics; ΔPPL paired by `sample_idx`; `per_gb_pass=None` when
    both configs skipped; realized-seq analytic basis; split `n_kv16/kv4_samples`.
  - R2 (1 **high** + 1 medium): **the high was a real correctness bug** — the prompt builder
    trimmed the already-templated token sequence, stripping the assistant generation suffix so
    the model would continue an unfinished user turn instead of summarizing (silently
    invalidating the whole experiment). Fixed: trim the transcript *body* and re-apply the
    chat template (verified: hits exact target tokens with `<|im_start|>assistant` suffix
    preserved). Medium: force `per_gb_pass=False` on `memory_inconclusive`.
  - R3 (1 low): a both-skipped length no longer drags a clean result down from
    PASS_WITH_ASTERISK to PARTIAL.
- **Total findings addressed: 12 (plan) + 8 (code, incl. 1 high) = 20.**

## Run status / blocker
- Code is complete and verified (ruff clean; prompt-builder and verdict-logic unit-checked).
- **Measurement sweep NOT yet run** — free RAM 10.9 GB < ~18 GB needed for the MLX-wired 30B.
  This is the documented [machine-contention blocker], identical to E1/E1b. The sweep is an
  idle-machine experiment by design; the script's preflight aborts rather than thrash.

## Next steps (on a quiesced machine — quit Cursor/Chrome/extra Claude procs)
1. `uv run python scripts/h9_e2_kv_workload.py smoke` — kv4 compatibility gate (asserts the
   quantized path is engaged, not a no-op).
2. `uv run python scripts/h9_e2_kv_workload.py run` — full sweep (3 lengths × {kv16,kv4} × 5).
3. Fill in findings + verdict here; close the issue / update the PR with measured numbers.
