# H3: Prompt-Aware Weight Pre-Loading

**Status**: Untested
**Analogy**: Database query planning / query optimization
**Bottleneck addressed**: Memory capacity (working set optimization)

## The Insight

A database doesn't load every table into memory for every query. The query planner analyzes the SQL, determines which tables and indexes are needed, estimates cardinality, and creates an execution plan that minimizes I/O.

LLM inference does the opposite: it loads the entire model (or hopes the right experts are cached) regardless of the prompt. But different prompts activate very different parts of the model:
- Code generation heavily uses certain experts/layers
- Creative writing activates different ones
- Math reasoning has its own "hot" components

## Hypothesis

> By analyzing the prompt before running full inference, we can predict the model's working set (which experts, layers, or weight subsets will be most active) and pre-load them into RAM, achieving higher cache hit rates and lower latency.

## Mechanism

```
1. CLASSIFY: Run prompt through a tiny classifier (<100M params)
   → Output: workload category (code, math, creative, chat, etc.)

2. PLAN: Look up the pre-computed "execution plan" for that category
   → Which experts are hot for this category
   → Which layers need full precision
   → Estimated memory budget

3. PRE-LOAD: Before starting inference, load the predicted working set
   → Priority-order expert loading from SSD
   → Evict cold experts from previous workload

4. EXECUTE: Run inference with warm cache
   → Most expert loads are cache hits
   → Fall back to on-demand loading for misses
```

## Why This Might Work

MoE expert activation patterns are known to cluster by task type. Research on expert specialization shows:
- Some experts "own" specific domains (code, language, reasoning)
- Expert activation is partially predictable from input features
- System prompts / few-shot examples strongly predict the overall workload type

On a 24GB machine, we can fit maybe 30-40% of a large MoE model's experts in RAM. If we load the **right** 30-40%, cache hit rates could jump from ~70% to ~90%+.

## Expected Impact

- **Cache hit rate**: 70% → 85-95% for categorized workloads
- **Latency reduction**: Fewer SSD reads during generation
- **No quality impact**: We're loading the same weights, just preemptively

## Prior Art (Research Findings)

Multiple groups have worked on this. The building blocks all exist, but the **end-to-end pipeline** (prompt text → classifier → expert loading plan → inference) is an **unexplored combination**.

### Pre-Inference Expert Prediction Systems

- **[SiDA-MoE (MLSys 2024)](https://arxiv.org/abs/2310.18859)**: Closest to our idea. Offline-trained **LSTM hash function** predicts which experts a batch will activate. Runs in a shadow thread concurrent with the previous batch. **3.93x throughput increase**, 72% latency reduction, 80% GPU memory savings, 99%+ prediction accuracy. Key: sparse activation means most experts never activate for a given input type.
- **[DAOP (DATE 2025)](https://arxiv.org/abs/2501.10375)**: Per-sequence activation patterns predict experts one layer ahead. Non-critical experts offloaded to CPU; predicted experts speculatively pre-calculated on CPU. **Unquantized Mixtral-8x7B: >4.5 tok/s on single A6000**. Key insight: CPU is idle during GPU compute — use it for speculative pre-computation.
- **[eMoE (2025)](https://arxiv.org/abs/2503.06823)**: Most explicitly "prompt-aware" system. Observes recurring patterns in token-to-expert routing across prompts. Predicts experts **every few prompts** (not every token), because adjacent prompts from the same task use the same experts. **80% memory reduction**, 17% latency reduction. Key finding: coarse-grained periodic prediction is valid — same cached experts work for several consecutive prompts.

### Expert Specialization Evidence

- **["What Gets Activated" (2026)](https://arxiv.org/abs/2601.10159)**: Analyzed three MoE LLMs across three domains. Identified two classes: **domain experts** (preferentially activated for specific domains, lower routing entropy) and **driver experts** (causally influential regardless of domain). Tokens earlier in sentences trigger driver experts more. **Direct evidence that prompt domain → expert activation mapping exists and is identifiable.**
- **[Input Domain Aware MoE (ACM MM 2025)](https://arxiv.org/abs/2510.16448)**: Routing using probabilistic mixture model trained independently of task objectives. Experts develop stable, interpretable specialization boundaries when routing is decoupled from task training.
- **[Stanford CS231N 2024](https://cs231n.stanford.edu/2024/papers/do-experts-specialize-a-mechanistic-exploration-of-mixture-of-ex.pdf)**: Sequence-level gating groups experts by topic/discourse. Token-level gating aligns with syntactic categories (nouns vs verbs). Specialization is more syntactic/structural at token level, more semantic at sequence level.

### Temporal Locality and Activation Tracing

- **[MoE-Infinity (2024)](https://arxiv.org/abs/2401.14361)**: Sequence-level expert activation tracing. **Only 3-20% of experts activate at a time, and 30-46% of activated experts are reused within a single sequence**. This temporal locality is the foundation for all prefetching. **4-20x latency reduction** vs baselines.

### Cache Replacement Strategy

- **[SpecMD (2026)](https://arxiv.org/abs/2602.03921)**: MoE expert access does NOT follow temporal locality (LRU/LFU assumptions fail). Access is *periodic and predictable*. "Least-Stale" eviction policy: **85x fewer collision misses** vs LRU, **88%+ hit rates at only 5% VRAM**.

### Key Numbers

| System | Prediction Method | Key Result |
|--------|------------------|------------|
| SiDA-MoE | LSTM hash, shadow thread | 3.93x throughput, 99% accuracy |
| DAOP | Per-sequence patterns | >4.5 tok/s on 90GB unquant model |
| eMoE | Prior distribution, periodic | 80% memory reduction |
| MoE-Infinity | Activation tracing | 4-20x latency reduction |
| SpecMD | Least-Stale eviction | 85x fewer misses vs LRU |

### The Gap

The specific pipeline — **classify prompt domain from raw text → map to predicted expert activation set → pre-load before any inference token is processed** — has not been built. The building blocks exist:
1. Domain classifiers (trivial with any small model)
2. Expert specialization maps (arXiv 2601.10159 shows they're identifiable)
3. Expert offloading infrastructure (MoE-Infinity, DAOP, eMoE)

No paper connects them end-to-end as a pre-inference planning step, analogous to how a database query optimizer uses cardinality estimation to plan page fetches before execution.

## Open Questions

1. ~~How many distinct workload categories are needed?~~ The expert specialization literature suggests domain-level (code, math, creative) is meaningful, plus syntactic categories at token level.
2. ~~How stable are expert activation patterns within a category?~~ **eMoE confirms**: adjacent prompts from same task use same experts — coarse-grained is fine.
3. Can we use the system prompt alone (often available before user input) to pre-load?
4. What's the switching cost when the workload changes mid-conversation?
5. ~~How do we build the category → expert mapping?~~ **arXiv 2601.10159** provides the method: entropy-based and causal-effect metrics on activation traces.
6. Can we combine SiDA-MoE's shadow-thread predictor with SpecMD's Least-Stale eviction?

## Experiment Plan

### Phase 1: Expert activation profiling
- Run diverse prompts (code, math, creative, chat) through an MoE model
- Record per-expert activation frequency for each category
- Measure: how distinct are the activation profiles? Is there meaningful clustering?

### Phase 2: Classifier + lookup table
- Build a lightweight prompt classifier (or use embeddings + k-NN)
- Map each class to a ranked list of expert priorities
- Measure: how many top-K experts from the predicted set actually get activated?

### Phase 3: Pre-loading pipeline
- Implement category-based expert pre-loading before inference starts
- Measure: cache hit rate improvement, time-to-first-token impact, overall tok/s

### Phase 4: Adaptive refinement
- Update the lookup table online as we observe actual activations
- Test: does the system improve over multiple conversations?

## Risks

- Expert activation may not cluster cleanly by prompt type
- The classification step adds latency to time-to-first-token
- Mid-conversation topic changes may thrash the cache
- Building good category profiles requires extensive offline profiling
