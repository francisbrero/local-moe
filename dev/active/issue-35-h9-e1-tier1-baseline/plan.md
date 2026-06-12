# H9-E1: Baseline Tier 1 Agentic Stack — Plan

**Issue:** #35 | **Branch:** `experiment/h9-e1-tier1-baseline`

## Hardware Target (reconciliation — plan-review HIGH finding)

`CLAUDE.md` opens with "16GB M4 MacBook Pro (~10-11GB usable)", but the **actual machine is a 24GB
M4 Pro**: every record in `experiments.jsonl` logs `chip="Apple M4 Pro", memory_gb=24.0`, and the
project memory (MEMORY.md) states 24GB M4 Pro. The H9 doc, the issue gates (≤17 GB RSS), and all
H0–H8 work assume 24GB. The 16GB line in CLAUDE.md is **stale** — a 30B-A3B 4-bit model (~17 GB)
cannot fit in 10–11 GB usable at all, so the ≤17 GB gate is only meaningful on 24GB.

Resolution: this experiment runs on, and gates against, the **24GB M4 Pro** (consistent with the
issue and all prior experiments). The stale CLAUDE.md hardware line is corrected as part of this
work. No 16GB re-validation is in scope — the H9 reframing is explicitly a 24GB story; 16GB would
require Tier 3 SSD-streaming (E5), not this Tier 1 baseline.

## Hypothesis

Qwen3-30B-A3B (3B active) at 4-bit on MLX clears all three Tier 1 gates on a 24GB M4 Pro:
≥20 tok/s decode, ≥90% structured tool-call accuracy, ≤17 GB steady-state footprint. H0 already
validated the MoE architecture on this hardware; this experiment confirms the *served stack* meets
the agentic-ops workload requirements and produces the reference numbers E2–E5 build on.

## Goal

Establish a measured baseline for the H9 Tier 1 stack: Qwen3-30B-A3B 4-bit serving
real agentic tasks on the 24GB M4 Pro. Gates all subsequent H9 experiments (E3, E5 depend on it).

## Success Gates (from issue)

- [ ] Decode speed ≥ 20 tok/s
- [ ] Tool-call success rate ≥ 90% on 20-case harness
- [ ] Peak RSS ≤ 17 GB steady-state

## Environment Findings (Phase 1)

- `mlx_lm` 0.31.1 installed; MLX core works on GPU. `vllm-mlx` NOT installed.
- **Qwen3-30B-A3B-4bit already in HF cache** (`mlx-community/Qwen3-30B-A3B-4bit`) — no ~17GB download.
- Disk: ~19 GB free — tight; avoid large extra downloads.

## Runtime Decision (user-directed)

- **Install vllm-mlx** per the H9 stack doc (issue specifies it). Treat as best-effort.
- **Primary gated baseline = vllm-mlx endpoint if it installs and serves.** Per plan-review
  finding (medium/feasibility): if vllm-mlx loads `mlx-community/Qwen3-30B-A3B-4bit` and serves a
  request, run the 20-case harness AND the perf measurements against its OpenAI-compatible endpoint
  over HTTP — this captures server overhead, request handling, and the runtime's own tool-call
  formatting. These are the **gated** numbers.
**Gating contract (definitive — resolves the round-2 vs round-4 tension):** there are two distinct
"baselines" and they serve different purposes:
- **`served` tier** = vllm-mlx HTTP endpoint. This is the ONLY tier that satisfies issue #35's
  goal and the ONLY tier that unblocks the E3/E5 served-stack dependency.
- **`engine` tier** = in-process mlx_lm. A valid measurement of the MLX *kernels* (so the experiment
  always yields concrete gate numbers and is never empty), but it is **explicitly INSUFFICIENT to
  unblock E3/E5** — those experiments depend on the served stack and must wait for a `served`-tier
  result (or an explicit issue amendment). An `engine`-tier pass is reported as "engine baseline
  established; served baseline still required."

So:
  - If vllm-mlx serves → vllm-mlx numbers are **gating**; mlx_lm in-process numbers are logged as
    **non-gating diagnostics** (the user-directed measurement instrument, useful as a kernel-level
    comparison and a sanity check on the endpoint numbers).
  - If vllm-mlx **fails to install or serve** → the *served-stack* baseline is recorded as **BLOCKED**
    (`status="blocked"`, failure mode documented). To avoid the experiment yielding **zero gated
    numbers** (per plan-review round-4 finding), the in-process mlx_lm path then produces a
    **gated baseline tagged `gating=true, gate_tier="engine"`** — i.e. it DOES evaluate
    issue #35's gates (so the experiment is never empty), but per the contract above is INSUFFICIENT
    to unblock E3/E5 — it does NOT exercise the served HTTP/server-process surface. This is the opposite of a silent
    substitution: the record's `runtime`, `gate_tier` (`served` vs `engine`), and `status` fields
    make the distinction loud. context.md + the PR state that E3/E5 either build on the engine
    baseline knowingly or wait for a `served`-tier vllm-mlx baseline.
  - When vllm-mlx **does** serve, its numbers are `gating=true, gate_tier="served"` and the mlx_lm
    numbers are `gating=false` diagnostics (a kernel-level sanity comparison).
- mlx_lm tool-calling is via `tokenizer.apply_chat_template(..., tools=[...])` (standard HF; Qwen3
  ships a tool-calling template) — **pre-flight confirmed available** in mlx_lm 0.31.1, so the engine
  gated path is never simultaneously blocked with vllm-mlx. The open risk is only whether *vllm-mlx*
  forwards OpenAI `tool_calls` correctly, caught by the Phase A (b) smoke test.
- Every result record carries `runtime` (`vllm-mlx` | `mlx_lm`), `gating` (bool), `gate_tier`
  (`served` | `engine` | null), and `status` (`completed` | `blocked`) so gate evaluation is
  unambiguous and no silent substitution occurs.

## Phases

### Phase A — Runtime setup
1. Attempt `uv add vllm-mlx` (or pip install in the venv). Record version or failure mode.
2. If installed, run a **two-stage smoke test** against the served endpoint:
   - (a) plain `/v1/chat/completions` completion loads `mlx-community/Qwen3-30B-A3B-4bit` and returns text;
   - (b) **tool-call smoke test**: a trivial one-function schema (e.g. `get_weather(city: str)`) and a
     prompt that should call it; require a structurally valid `tool_calls` response.
   vllm-mlx is treated as the gated runtime **only if BOTH (a) and (b) pass**. If (b) fails (no/broken
   OpenAI tool_calls support), the vllm-mlx baseline is marked **BLOCKED** even if completions work.
3. Confirm in-process mlx_lm load of the same model works (`mlx_lm.load`).

### Phase B — Tool-call harness (20 cases)
Build `scripts/h9_e1_baseline.py` + `scripts/h9_e1_harness.py` (cases as data).
20 cases across the real task types named in the issue:
- Email triage / label-route (e.g. 6 cases)
- Calendar lookup (4 cases)
- Slack post (4 cases)
- Short summarization (3 cases — `expected_tool: null`: correct behavior is to answer directly, no tool)
- Mixed / multi-arg (3 cases)

Each case: `{id, category, user_msg, tools (JSON schema list), expected_tool, validators}`,
where `expected_tool` is either a tool name or **`null`** for no-tool cases, and `validators` is a
per-case callable (`args_dict -> bool`) asserting key argument *values* (recipient, channel, date,
enum membership). No-tool cases set `expected_tool: null` and omit `validators`.
Grading splits by case type and requires **semantic** correctness, not just parseable structure
(per plan-review finding — a call to the right tool with the wrong recipient/channel/date is a fail):
- **Tool-call cases** (`expected_tool` set): pass requires ALL of —
  1. structurally-valid tool call (parses, correct envelope) — tracked as a `structural_valid` submetric;
  2. tool **name** matches `expected_tool`;
  3. required args present and well-typed;
  4. a **per-case `validators` function** passes, checking key argument *values*: e.g. correct email
     recipient/label, correct Slack channel, date/time normalized to the expected value, enum values
     in the allowed set. Each case in `h9_e1_harness.py` ships its own `validators` (not a generic
     presence check). The **≥90% gate is computed on this semantic pass**; `structural_valid` is
     reported alongside as a submetric so "parses but wrong value" is visible.
- **No-tool cases** (`expected_tool: null`, e.g. some short-summarization prompts where the right
  behavior is to answer directly): pass = model does NOT emit a tool call (answers in text). These
  are graded separately and the per-category breakdown is reported, so a model that over-calls tools
  is distinguishable from one that under-calls. The overall ≥90% rate is computed across all 20 cases.
Record malformed outputs and whether a lenient recovery parser (Rapid-MLX-style) would have rescued them.

The harness is **runtime-agnostic**: cases + grader live in `h9_e1_harness.py` as pure data +
functions. A thin adapter layer exposes `run_case(case) -> raw_output` for each backend:
- **vllm-mlx adapter**: POST to the **OpenAI-compatible `/v1/chat/completions`** with `tools=[...]`,
  read `tool_calls` from the response.
- **mlx_lm adapter**: use mlx_lm's chat template tool-calling support and parse the emitted call.
Same cases, same grader, both backends — so the comparison is apples-to-apples.

**Gated API shape (committed, per plan-review finding):** the gating contract is the **OpenAI
`/v1/chat/completions` + `tools` / `tool_calls`** format. Tool schemas are written as OpenAI
function-tool JSON. (vllm-mlx also speaks the Anthropic Messages API, but the OpenAI shape is the
single gated surface — chosen because mlx_lm's tool-calling and most agent frameworks target it.)
Each tool-call response is normalized to `{name, arguments}` by the adapter before grading, so the
grader never sees runtime-specific envelope differences.

### Phase C — Performance measurement
In `scripts/h9_e1_baseline.py`:
- **Decode tok/s — pinned gate protocol** (so E2–E5 compare cleanly):
  - Fixed prompt: ~512 tokens (verified tokenized length); `max_new_tokens = 256`; greedy (temp 0).
  - **Generated tokens only** in the rate (exclude prompt); rate = `(gen_tokens - 1) / (t_last - t_first)`
    so prefill/TTFT is excluded from the decode number.
  - **Non-streaming** request for the gate; N = 5 warm runs after 1 discarded cold run; report the
    **median** as the gated decode tok/s (also report cold tok/s separately).
  - **Served-mode timing path (concrete, per plan-review finding):** prefer vllm-mlx's response
    `usage` / timing fields if present. If the OpenAI-compatible endpoint does not expose prefill
    time or per-token timings (likely — plain OpenAI schema omits them), use a **streaming request**
    and time server-sent token deltas: TTFT = time to first chunk; decode rate from inter-chunk
    deltas of the steady-state tokens. Localhost transport overhead is sub-ms per token and is noted
    as a documented caveat (`timing_method` field records which path was used). prefill tok/s in
    served mode is derived as `prompt_tokens / TTFT` (TTFT ≈ prefill time for a cold KV at fixed
    prompt) — labelled approximate. If even streaming timings are unreliable, the field is recorded
    `null` with `timing_method="unavailable_served"` rather than reporting a misleading number.
- **TTFT**: time-to-first-token, recorded per generation, reported separately from decode rate.
- **Prefill tok/s @ 8K** and **@ 16K**: synthesize a prompt of the target token length, measure
  prompt-eval throughput (prompt tokens / prefill time), `max_new_tokens` small (e.g. 8).

**Memory methodology (per plan-review finding — unified memory is subtle):**
- Report memory as **separate rows per condition**, not one summed number:
  `idle_loaded` (model loaded, no generation), `decode_short`, `prefill_decode_8k`, `prefill_decode_16k`.
- **Primary gate metric = `phys_footprint` of the serving process tree** (via `proc_pid_rusage` /
  `psutil` USS-equivalent; on macOS use `task_info` phys_footprint if accessible, else process RSS).
  This is the single number compared against the ≤17 GB gate.
- **Process-tree accounting (per plan-review finding):** vllm-mlx runs a **server process** separate
  from the harness client and may spawn helper/child processes. Record the server PID(s), sum
  phys_footprint across the server's process tree, and **exclude the client harness** (it is a thin
  HTTP caller). The ≤17 GB gate applies to the **server process-tree peak during the 16K
  steady-state condition**. For the in-process mlx_lm diagnostic path there is a single PID (the
  harness *is* the model host) — noted so the two runtimes' footprints are interpreted correctly.
- Report **MLX Metal active/peak** (`mx.get_active_memory()` / `mx.get_peak_memory()`) as a
  *separate* column — do NOT sum it with RSS (Metal wired memory is part of the process footprint
  under unified memory; summing double-counts). If RSS and Metal-peak are verified disjoint we note
  it, otherwise phys_footprint is authoritative.
- **Served mode caveat (per plan-review finding):** `mx.get_*_memory()` reads the *calling* process,
  so in served mode the harness cannot read the vllm-mlx server's MLX counters. For `served` tier,
  Metal active/peak is recorded `null` (`metal_method="server_process_inaccessible"`) and the
  phys_footprint of the server process tree is the authoritative memory number. For `engine` tier
  (single process) both phys_footprint and Metal counters are available.
- The **gate applies to the 16K-context steady-state row** (worst case: full model + max KV cache).
  Qwen3-30B-A3B KV at 16K ≈ 1.5 GiB before allocator overhead — explicitly accounted for.
- **Headroom reporting (per plan-review finding):** alongside the ≤17 GB gate, report measured
  **available headroom against the 19–21 GB practical budget** of the 24GB machine. This lets a
  gate failure distinguish "missed the aggressive Tier 1 17 GB target" (still operational on 24GB,
  e.g. 17–20 GB) from "exceeds the machine's usable envelope" (>~20 GB, genuinely unviable). Both
  the gate result and the headroom verdict are reported.
- Sampling: background thread samples phys_footprint + Metal-active every 200ms during each
  condition; report the peak per condition.
- **Measurement-method fallback (per plan-review finding):** the gate is defined against macOS
  **phys_footprint** when obtainable, else **RSS** — and the record's `mem_method` field
  (`phys_footprint` | `rss`) states which was used so the gate's meaning is never silently changed.
  Collection order, no root/dtrace required:
  1. macOS-specific phys_footprint via `ctypes` → `proc_pid_rusage(pid, RUSAGE_INFO_V2)` reading
     `ri_phys_footprint` (the same number Activity Monitor's "Memory" column shows). Works on other
     PIDs owned by the same user — so it covers the vllm-mlx server tree.
  2. If that call fails, `psutil.Process(pid).memory_info().rss`.
  Walk `children(recursive=True)` and sum across the server tree; `mem_method` reflects the path
  actually taken (and is consistent across the tree).

### Phase D — Log + report
- Log structured records to `experiments.jsonl` via `log_experiment`
  (experiment_name `h9_e1_baseline`, phases `tool_calls`, `perf_decode`, `perf_prefill`).
- Each record `config` carries `runtime` (`vllm-mlx`|`mlx_lm`), `gating` (bool), `gate_tier`
  (`served`|`engine`|null), and `status` (`completed`|`blocked`).
- Perf records include `ttft_ms` (repo convention). **Quality** is represented by the tool-call
  success rate, not perplexity.
- For consistency with the repo's `experiments.jsonl` convention, `perplexity` and `cache_hit_rate`
  are included in `results` as **`null` with a sibling reason field** rather than silently omitted:
  `perplexity: null, perplexity_na_reason: "quality represented by tool-call rate; 4-bit PPL covered by H7 (PR #20)"`,
  `cache_hit_rate: null, cache_hit_rate_na_reason: "resident model, no SSD expert streaming; cache hits measured in H0 (PR #16)"`.
  Keeps records schema-consistent and makes the intentional gaps + their prior coverage explicit.
- Write findings + gate pass/fail table to `context.md`.

## Measurement Methodology Notes

- **tok/s**: use mlx_lm's generation metadata (it reports prompt/gen tokens + timings) when
  available; otherwise wall-clock around `generate` with token counts from the tokenizer.
- **Warm-up**: discard the first generation (kernel compilation / cold cache) from warm numbers.
- **Determinism**: temperature 0 / greedy for tool-call grading to make pass/fail reproducible.
- **Context lengths**: build 8K/16K prompts from repeated realistic transcript text, verify the
  actual tokenized length is within ±5% of target before measuring.
- **Steady-state RSS**: measure with the model loaded and a generation in flight; the model is the
  only large process (we do not need to literally open a browser — document this as the measured
  condition vs. the issue's "alongside browser + Slack" framing).

## Baseline Commands

```bash
# Setup
uv add vllm-mlx            # best-effort; or: uv pip install vllm-mlx

# vllm-mlx serve (primary, if installed) — exact flags confirmed at impl time
uv run python -m vllm_mlx.serve --model mlx-community/Qwen3-30B-A3B-4bit   # placeholder; verify CLI

# Baseline. --runtime auto = use vllm-mlx if it serves (GATED), else also run mlx_lm in-process
# but ONLY as non-gating diagnostics — the mlx_lm fallback CANNOT satisfy issue #35's gates.
uv run python scripts/h9_e1_baseline.py --runtime auto --context 8k,16k
uv run python scripts/h9_e1_harness.py  --runtime auto
```

## Rollback / Fallback

- **vllm-mlx fails to install or serve** → vllm-mlx baseline recorded as **BLOCKED**
  (`status=blocked`, `gating=false`); in-process mlx_lm numbers logged as **non-gating diagnostics**
  only (NOT substituted as the gated baseline). Document the failure mode in context.md and flag
  that E3/E5 await a real vllm-mlx baseline (or an issue amendment).
- **A gate fails** (e.g. decode < 20 tok/s, RSS > 17 GB, tool-call < 90%) → this is a valid
  negative result. Log it, report it honestly in context.md and the PR; do not tune top-k/kv_bits
  to chase the gate (those are E4/E2). The baseline number stands as the reference.
- **16K context OOMs / pages out heavily** → record the failure + the max context that fits, report
  as a finding; the ≤16K-usable-context requirement becomes a measured limit rather than an assumption.

## Out of Scope

- E2 (KV quant at long context), E3 (two-tier router), E4 (top-k sweep), E5 (Tier 3) — separate issues.
- We do NOT tune top-k or kv_bits here; baseline uses model defaults so later experiments have a clean reference.

## Deliverables

- `scripts/h9_e1_baseline.py` — perf measurement + orchestration
- `scripts/h9_e1_harness.py` — 20 tool-call cases + grader
- New `experiments.jsonl` records
- `context.md` with gate results
