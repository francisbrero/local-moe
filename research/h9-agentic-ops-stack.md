# H9: The Agentic Ops Stack — Largest Useful Model for Daily Knowledge Work

**Status**: Proposed (synthesis of internal H0–H8 findings + external state-of-the-art survey, June 2026)

## The Goal, Reframed

The previous hypotheses (H0–H8) optimized for "largest model that runs at all" — culminating in 72B dense streaming at 0.007 tok/s (H8a), with a best-case projection of 0.5–2 tok/s (H8b). That is batch-only territory.

This document reframes the target around the actual workload: **daily agentic ops** — answering email, triaging Slack, prioritizing features, summarizing call recordings to prep meetings. These tasks need:

| Requirement | Why | Hard number |
|---|---|---|
| Interactive decode speed | Multi-turn agent loops, drafting | ≥ 20 tok/s |
| Reliable tool calling | MCP/function calls for email/Slack/calendar | ≥ 90% call accuracy |
| Long context | Call transcripts (30–60 min ≈ 8–15K tokens), email threads | 16–32K usable context |
| Fast prefill | Summarization is prefill-dominated | ≥ 300 tok/s prompt |
| Memory headroom | Runs alongside browser, Slack, etc. | ≤ 8–10 GB co-resident footprint (phys_footprint, not RSS — revised post-E1) |

A 72B dense model at 0.5 tok/s fails every row. A 30B MoE with 3B active parameters passes all of them — and our own H0 experiments already validated exactly this architecture class on this hardware.

## Why MoE-with-mmap Wins (Internal + External Evidence Converge)

Our H0 work proved the architecture on a 24GB M4 Pro:
- Non-expert weights of Qwen3-30B-A3B: **0.5 GB at 4-bit** (fits trivially)
- Expert streaming latency p50 0.40ms; streamed experts ran *faster* than in-memory in Phase 4a
- OS page cache hits 63–78% under Zipf routing patterns ("trust the OS" confirmed)
- GPU/SSD concurrency costs only 0.2% — I/O and compute fully overlap on unified memory

The June 2026 external survey confirms the ecosystem caught up to the same conclusion:
- **MLX decisively won on Apple Silicon** — 1.4–3x faster than llama.cpp for MoE generation; Ollama 0.19 switched its Mac backend to MLX
- Qwen3-30B-A3B Q4 runs at **30–50 tok/s on M4 Pro 24GB** under MLX; ~17 tok/s via llama.cpp mmap with only ~3 GB resident
- **Tool-calling quality is training-dependent, not size-dependent**: Qwen3.5-4B scores 97.5% on tool-call benchmarks, beating models 10x larger; GLM-4.7-Flash (30B/3B MoE) scores 95% at ~52 tok/s
- SwiftLM demonstrates SSD-streamed Qwen3.5-122B at 5.2 tok/s with speculative prefetch (~70% cache hit) — the production version of our H0/H2 ideas

Key insight: **a 30B MoE beats a 14B dense at the same memory budget** — similar speed (3B vs 14B active), much higher quality. MoE sparsity is the multiplier that makes "significantly powerful and large" compatible with 24GB.

## Recommended Stack

### Tier 1: Qwen3-30B-A3B / Qwen3.5-35B-A3B on MLX

> **Revised post-E1**: MLX wires the full 16.55 GB (phys_footprint) and cannot coexist with a normal working set on 24 GB — reproduced 4×. This tier is therefore the **off-hours/batch** configuration (machine idle, ~19–21 GB usable). The **concurrent daily-driver** candidate is the same 30B in llama.cpp mmap mode (~3 GB resident, OS pages experts) or Tier 2's 4B — decided by E1b below.

| Component | Choice | Rationale |
|---|---|---|
| Model | **Qwen3-30B-A3B** 4-bit (~17 GB), or Qwen3.5-35B-A3B if it fits with context | Best quality/speed/tool-calling balance in every 2026 benchmark |
| Alternative | **GLM-4.7-Flash** (30B/3B, ~18 GB) | Top tool-call scorer (95%) at ~52 tok/s |
| Runtime | **vllm-mlx** (MCP + Anthropic Messages API, continuous batching) or **Rapid-MLX** (17 tool-call parsers with malformed-output recovery) | Both serve OpenAI-compatible endpoints for any agent framework |
| KV cache | `kv_bits=4, kv_group_size=64` for contexts > 4K | **Our H7 result**: zero PPL loss, 3.56x compression — call transcripts are exactly this regime |
| Routing | `--moe-top-k 5` (down from 8) | +7–16% speed, no measurable quality loss (external benchmarks; matches our H0 routing analysis) |
| Expected | **30–50 tok/s decode, ~12–17 GB resident** (less with mmap mode) | Passes every workload requirement above |

### Tier 2 (fast path): Qwen3.5-4B router/worker

Run Qwen3.5-4B (3.4 GB, 120+ tok/s, 97.5% tool-call accuracy) alongside Tier 1 as the **triage layer**: classify incoming email/Slack, execute simple tool calls, route only drafting/reasoning/summarization to the 30B. Both fit in memory simultaneously (~21 GB worst case; use mmap mode for the 30B to keep resident set low). This fast/slow split mirrors how the workload actually distributes — most triage decisions are cheap.

### Tier 3 (stretch, where our research compounds): SSD-streamed 100B+ MoE

SwiftLM streams Qwen3.5-122B (22B active) on 64GB machines at 5.2 tok/s. On 24GB this is currently out of reach with their resident-set assumptions — but our H0 + H5 + H2 results suggest a path:
- H5 sensitivity-guided Q2/Q3 on expert weights (experts tolerate aggressive quantization; protect router + last blocks at Q4+) shrinks the streamed working set toward page-cache size
- H2 routing prediction (93–97% accuracy in literature, untested by us) replaces SwiftLM's ~70% speculative prefetch hit rate
- H0's measured 5–7 GB/s scattered reads + full GPU/SSD overlap set the bandwidth ceiling

This is the genuinely novel contribution left in this repo: **nobody has combined per-expert sensitivity-guided 2-bit quantization with predictive prefetch on Apple Silicon.** Target: GLM-4.6-Air-class or Qwen3.5-122B-class model at ≥ 5 tok/s on 24GB. Useful for overnight/batch jobs (deep call-recording analysis, weekly priority reviews) even if not interactive.

## Workload → Configuration Map

| Daily task | Model tier | Critical config |
|---|---|---|
| Email triage/labeling | Tier 2 (4B) | Tool calls only; speed matters |
| Email drafting | Tier 1 (30B) | Standard context |
| Slack triage | Tier 2 (4B) | High volume, low complexity |
| Feature prioritization | Tier 1 (30B) | Reasoning-heavy; consider DeepSeek-R1-Distill-14B as alternate |
| Call summarization / meeting prep | Tier 1 (30B) | **kv_bits=4 essential** (8–15K token transcripts); prefill-dominated |
| Deep/batch analysis | Tier 3 (122B streamed) | Overnight; quality over latency |

Agent harness: any OpenAI-compatible framework (PydanticAI, Hermes Agent, LangChain) pointed at the local endpoint; vllm-mlx speaks MCP natively for the email/Slack/calendar connectors.

## Experiment Plan (original — superseded by the Plan Revision below after E1)

**E1 — Baseline the Tier 1 stack (1 day).** Install vllm-mlx + Qwen3-30B-A3B-4bit. Measure: decode tok/s, prefill tok/s at 8K/16K context, RSS, tool-call success rate on a 20-case harness mirroring real tasks (email triage, calendar lookup, Slack post). Gate: ≥ 20 tok/s decode, ≥ 90% tool calls, ≤ 17 GB peak RSS.

**E2 — KV quantization at workload context lengths (0.5 day).** Re-run H7's kv4 config on real call-transcript summarization (10–15K tokens). We only validated short contexts; the external claim is that overhead amortizes past 4K. Gate: summarization quality holds, ≥ 1.5x effective context per GB.

**E3 — Two-tier router (1–2 days).** Qwen3.5-4B triage + 30B worker behind one endpoint. Measure end-to-end latency on a mixed task stream and peak combined RSS. Gate: p50 triage < 2s, no memory pressure (pageouts < 200 MB/hr).

**E4 — moe-top-k and mmap-mode trade-offs (0.5 day).** top-k ∈ {8,6,5,4} quality/speed curve on our task harness; MLX-resident vs llama.cpp-mmap resident-set comparison for the "Mac is doing other things" scenario.

**E5 — Tier 3 feasibility spike (3–5 days, after E1–E3 ship).** Apply H5 OptiQ sensitivity analysis to a large MoE's experts (open question from H5: does the method transfer from dense to MoE?), produce mixed Q2/Q4 expert shards, stream via the H8a safetensors mmap path, add H2 next-layer routing prediction for prefetch. Gate: ≥ 2 tok/s on a ≥ 100B-total-parameter MoE; stretch ≥ 5 tok/s.

## E1 Findings & Reframe (June 2026, issue #35 / PR #40) — for researcher review

E1 ran on the actual 24 GB M4 Pro (vllm-mlx 0.3.0 + mlx_lm 0.31.1, Qwen3-30B-A3B-4bit). It surfaced a
result that, we think, reframes the whole Tier 1 question. **Flagging for direction — we have not acted
on this beyond notes.**

### What we measured (solid)
- **vllm-mlx 0.3.0 installs and serves** the model with an OpenAI-compatible endpoint and tool-calling
  (`--enable-auto-tool-choice --tool-call-parser hermes`). The served stack works.
- **Idle model footprint = 16.55 GB**, measured via macOS `phys_footprint` (`proc_pid_rusage`
  `RUSAGE_INFO_V2`). **Standard RSS reports only ~0.37 GB** — it does not count MLX's wired Metal
  memory. *Takeaway: RSS is the wrong gate metric for MLX; phys_footprint is authoritative.* The
  "≤ 17 GB peak RSS" gate in E1 should be read as phys_footprint, and the model is already at 16.55 GB
  idle (i.e. it essentially fails the gate before adding KV cache).

### The blocker, and why it matters more than the gate
- We could **not** obtain decode tok/s, prefill, or the tool-call success rate. Reason: the model
  loads to 16.55 GB, but the dev machine's normal working set (Cursor, Chrome, Xcode, several Claude
  processes — ~4 GB+ RSS, more wired) means 16.55 GB + apps **exceeds physical RAM**. macOS enters
  sustained swap (`vm.swapusage` 16–19 GB used, free RAM ~12%) and `mlx_lm.load()` never reaches a
  usable state — worker pinned at 0–25 % CPU, I/O-bound. **Reproduced 4× across both runtimes**, at
  starting-free-RAM from 9.8 to 16.6 GB. With nothing else running it loads fine; the failure is
  co-residency, not capacity.

### The reframe (this is the part we want input on)
The model is **not too big to fit** (16.55 < 24 GB). It is **too big to coexist** with the user's real
workload. The binding constraint for a *daily local agent* is not "largest model that fits on the
laptop" — it is **"largest model that fits in the headroom the user isn't already using,"** realistically
6–10 GB on a machine someone is actually working on. This suggests Tier 1 is really two problems:

1. **Off-hours / batch tier** — machine idle, ~19–21 GB usable. Home of the big MoE (and Tier 3
   SSD-streamed 100B+). Latency irrelevant; gates are fit + correctness. The 30B-A3B at 16.55 GB
   belongs *here*, not in the interactive daily-driver slot.
2. **Concurrent / co-resident tier (the real daily-agent challenge)** — machine busy. Gate is the
   model's *resident working set fits the leftover headroom and does not force pageouts under
   realistic pressure*. A 16.55 GB MLX-wired model does not qualify on 24 GB; candidates are smaller
   (Qwen3.5-4B ~3.4 GB) or the same 30B in **llama.cpp mmap mode** (this doc cites ~3 GB resident — the
   OS pages experts via page cache instead of wiring all 16.55 GB). E1 used MLX, which wires everything,
   so the mmap-resident-set path is exactly what E1 did *not* exercise.

### Open questions for the researcher
- Should we split Tier 1 into "off-hours batch" vs "concurrent daily-driver" with **distinct memory
  budgets and gates**? If so, what's a defensible co-resident budget (we'd suggest gating against an
  8–10 GB memory-pressure floor, using the repo's `experiment_utils.create_memory_pressure`, as H8b did)?
- Is the right *interactive* gate **pageout rate under N GB of competing ballast** (e.g. < 200 MB/hr —
  same shape as E3's gate) rather than peak footprint?
- Does **llama.cpp mmap mode** actually hold the 30B-A3B at ~3 GB resident and survive co-residency
  where MLX-wired thrashed? This is the cheapest decisive probe and would tell us whether 30B is viable
  concurrent at all (currently scoped only as a sub-bullet of E4). Worth promoting?

## E1b Findings (June 2026, issue #41) — the mmap co-residency probe

E1b ran Qwen3-30B-A3B **Q4_K_M GGUF** (Qwen/Qwen3-30B-A3B-GGUF, 18.56 GB, sha 0d003f66…) via
`llama-server` (mmap default, `--jinja`), under an exactly-8 GB fixed ballast. It answers the
promoted question — *does llama.cpp mmap hold the 30B at low resident set and survive co-residency
where MLX-wired thrashed?* — and the answer is **a nuanced "the memory story changes, the speed
verdict doesn't (on this contended machine)."**

### The crux result: mmap resident set is ~4 GB, but phys_footprint reports ~18 GB
Idle, `-ngl 0` (CPU/mmap), no ballast:
- **phys_footprint = 17.91 GB** but **`ri_resident_size` = 4.01 GB.**

mmap works exactly as hoped — the OS keeps only ~4 GB of weights RAM-resident and pages the rest
from the file. **But `phys_footprint` over-counts mmap**: it charges the full ~18 GB of file-backed
pages to the process even though the OS can evict them under pressure. This is the **exact mirror of
E1**, where RSS *under*-counted MLX's wired memory (0.37 GB reported vs 16.55 GB real).

> **Methodology consequence:** phys_footprint is authoritative for *wired* memory (MLX) but
> *over*-states *mmap* memory. For the mmap concurrent tier, **`ri_resident_size` is the truer
> working-set metric.** By resident_size (~4 GB) the 30B **fits** the ≤ 8 GB concurrent budget;
> by phys_footprint (~18 GB) it does not. The E1b plan-review flagged this page-cache blind spot
> in advance.

### What was measured (all in experiments.jsonl)
- **Tool-call quality: 18/20 = 90.0%** (CPU/ngl-0, quiesced). Calendar/Slack/summarization/mixed
  all 100%; email triage 4/6. Qwen3-30B-A3B via llama.cpp is a competent tool-caller — **passes**.
- **Disk-read churn 0 MB/s steady-state** — once mmap'd in, weights are not re-faulted from SSD
  (the page cache holds them); the churn gate **passes** (per-process `ri_diskio_bytesread`,
  calibrated cold=64 MB vs warm=0 MB).
- **Decode under 8 GB ballast: not demonstrable** — CPU 30B under the dev machine's residual swap
  (13–28 GB) produced no tokens in the request window; pageouts 283 MB/hr (> 200). **Same host
  contention that blocked E1** — the machine could not be fully quiesced this session.
- **Full Metal offload (`-ngl 999`) is GPU-OOM-blocked even idle:** the 30B (~18 GB) + 8 GB prompt
  cache + KV exceeds the M4 Pro's **18186 MiB GPU budget** (`kIOGPUCommandBufferCallbackError
  OutOfMemory`). The usable Mac config is **partial offload** with the prompt cache trimmed.

### Verdict & implication for E3
- **Categorical win over E1:** llama.cpp mmap **loads** the 30B under contention; MLX could not.
- **Memory thesis: supported by the right metric.** Resident set ~4 GB fits the concurrent budget;
  the 30B is NOT off-limits the way MLX-wired 16.5 GB was.
- **Speed thesis: inconclusive** on this contended, 18 GB-GPU machine. The one missing number is
  partial-offload decode tok/s on a *genuinely quiesced* machine (scripts ready).
- **Recommendation:** 30B-via-llama.cpp-mmap is a **viable-but-unproven concurrent candidate**
  (memory ✓ by resident_size, quality ✓ at 90%, speed TBD). Keep ≤ 4B as the safe concurrent
  default until a clean partial-offload decode number clears ≥ 12 tok/s. E3 should benchmark
  partial-offload, not full Metal offload, and gate memory on **resident_size**, not phys_footprint.

## Plan Revision (post-E1 researcher response, 2026-06-12)

The E1 reframe is accepted. The binding constraint for a daily local agent is **headroom the user isn't already using**, not total RAM. Answers to the three open questions, then the revised plan.

### Decisions

1. **Yes — split Tier 1 into two operating modes with distinct budgets and gates.**
   - **Off-hours/batch mode**: machine idle, budget ~19–21 GB phys_footprint. Gates: fit + correctness + tok/s on an idle machine. Home of MLX-wired 30B-A3B (16.55 GB measured) and the Tier 3 streamed 100B+.
   - **Concurrent/daily-driver mode**: machine busy, budget **≤ 8 GB phys_footprint** under an **8 GB competing-ballast floor** (use `experiment_utils.create_memory_pressure`, as H8b did). 8 GB ballast on 24 GB leaves ~16 GB; gating the model at ≤ 8 GB preserves ~8 GB for the OS/page cache — and matches the observed 4 GB+ apps RSS plus wired overhead with margin.
2. **Yes — the interactive gate is behavior under pressure, not peak footprint.** Primary gates: sustained **pageouts < 200 MB/hr** and **p50 first-token latency** under ballast. Peak phys_footprint becomes a budget check, not the headline gate. **RSS is dropped from all gates** — E1 proved it blind to MLX wired memory (0.37 GB reported vs 16.55 GB actual). All experiments report phys_footprint via `proc_pid_rusage`.
3. **Yes — promote the llama.cpp mmap co-residency probe to its own experiment (E1b), run next.** It is the cheapest decisive test of whether any 30B is viable in the concurrent slot. If it fails, the concurrent tier defaults to ≤ 4B-class models and the 30B is off-hours only — still a coherent stack, just a sharper split.

### Revised experiments

**E1 — status: partial, blocked on co-residency (done).** Stack serves with tool-calling; footprint measured (16.55 GB wired); load failure under real workload reproduced 4×. Decode/prefill/tool-call metrics were not obtainable co-resident. The remaining idle-machine measurements move to E4.

**E1b — llama.cpp mmap co-residency probe (NEW, next, 0.5–1 day).** Qwen3-30B-A3B Q4_K_M GGUF via `llama-server` (mmap default), under 8 GB ballast from `create_memory_pressure`. Measure: phys_footprint, decode tok/s, p50 first-token latency, pageout rate over ≥ 30 min sustained generation, and the 20-case tool-call harness. Gate: ≥ 12 tok/s decode under ballast, pageouts < 200 MB/hr, phys_footprint ≤ 8 GB. **This experiment decides the shape of the concurrent tier.**

**E2 — KV quantization at workload context lengths (rescoped, 0.5–1 day).** Two arms now: (a) MLX `kv_bits=4` on the off-hours tier, idle machine, 10–15K-token call-transcript summarization; (b) llama.cpp `-ctk q8_0` (and `-ctv q4_0` if stable on Metal) on the concurrent tier under ballast — KV cache pages compete with expert pages in mmap mode, so this interaction is new territory. Gates unchanged: quality holds, ≥ 1.5x effective context per GB.

**E3 — Two-tier router (rescoped, 1–2 days).** Qwen3.5-4B always-resident (MLX-wired, ~3.4 GB) + 30B worker in whichever mode E1b selects (mmap concurrent, or MLX off-hours-only with the router queueing heavy jobs). Run the mixed task stream **under 8 GB ballast**. Gates: p50 triage < 2 s, pageouts < 200 MB/hr, combined phys_footprint ≤ 8 GB (4B wired + 30B resident set).

**E4 — top-k sweep + idle-machine MLX baseline (rescoped, 0.5 day).** `--moe-top-k` ∈ {8,6,5,4} quality/speed curve on the task harness, plus the E1 metrics that were blocked (decode/prefill tok/s, tool-call rate on MLX, idle machine) to characterize the off-hours tier. The mmap-vs-MLX comparison has been promoted to E1b.

**E5 — Tier 3 feasibility spike (unchanged scope, off-hours tier explicitly).** As originally specified; all memory gates in phys_footprint terms. Runs after E1b–E3.

### Standing methodology changes

- **phys_footprint replaces RSS** in every gate and in `experiments.jsonl` logging.
- **Every interactive-tier experiment runs under memory-pressure ballast**; idle-machine numbers are labeled off-hours and never quoted as daily-driver performance.

## What This Supersedes

- **H8b (72B dense Q2 streaming, issue #32)**: deprioritize. Even its best-case 2 tok/s is dominated by Tier 1 on every agentic-ops dimension, and Tier 3 (MoE streaming) is a strictly better use of the same H8a machinery — streaming 2–3 MB experts with 63–78% cache hits beats streaming 262 MB dense blocks with ~55% coverage.
- Custom TurboQuant work stays dead (H7 verdict); note the external mlx-optiq package now ships KV compression (−44% cache) if we ever need more than kv4.

## Sources

Internal: H0 (PR #16), H5 (PR #27), H7 (PR #20), H8a (PR #31), experiments.jsonl (77 runs). External survey (June 2026): SwiftLM (github.com/SharpAI/SwiftLM), vllm-mlx (github.com/waybarrios/vllm-mlx), Rapid-MLX (github.com/raullenchai/Rapid-MLX), llama.cpp MoE-offload PoC (ggml-org/llama.cpp discussion #23324), JD Hodges 2026 tool-calling benchmark (jdhodges.com), Ollama 0.19 MLX backend (ollama.com/blog/mlx), Apple ML Research M5 MLX study, NPUMoE (arXiv:2604.18788), MxMoE (arXiv:2505.05799).
