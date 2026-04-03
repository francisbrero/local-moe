"""
routing_trace.py — Phase 1: Expert routing pattern analysis for H2.

Hooks into MLX model inference to capture:
- Expert activation indices per layer per token
- Pre-attention hidden states (block-input residual stream)

Runs Qwen3-30B-A3B (4-bit) on diverse prompts and logs routing statistics.

Usage:
    uv run python scripts/routing_trace.py [--pilot] [--max-tokens 500]
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import psutil

from experiment_utils import get_environment_info, get_rss_mb, log_experiment

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MOE_REPO = "mlx-community/Qwen3-30B-A3B-4bit"
MEMORY_FLOOR_GB = 2.0
TRACE_DIR = Path(__file__).parent.parent / "routing_traces"

PILOT_PROMPTS = {
    "prose": (
        "Write a detailed essay about the history of computing, from Charles Babbage's "
        "analytical engine through modern GPUs and AI accelerators. Cover the key milestones, "
        "the people involved, and how each breakthrough built on the last."
    ),
    "code": (
        "Implement a B-tree data structure in Python with insert, search, and delete operations. "
        "Include docstrings and handle edge cases like node splitting and merging."
    ),
    "qa": (
        "What are the fundamental differences between TCP and UDP? When would you choose one "
        "over the other? Give specific examples from real-world applications."
    ),
    "reasoning": (
        "A farmer has a fox, a chicken, and a bag of grain. He needs to cross a river in a "
        "boat that can only carry him and one item at a time. If left alone, the fox will eat "
        "the chicken, and the chicken will eat the grain. How does the farmer get everything "
        "across safely? Explain your reasoning step by step."
    ),
}

FULL_PROMPTS = {
    **PILOT_PROMPTS,
    "prose_long": (
        "Describe the complete process of how a modern CPU executes a single instruction, "
        "from fetch through writeback. Include pipelining, branch prediction, OoO execution."
    ),
    "code_python": (
        "Write a complete async HTTP client in Python using asyncio with connection pooling, "
        "retry logic with exponential backoff, timeouts, and proper error handling."
    ),
    "code_algo": (
        "Implement a concurrent skip list in Python with lock-free reads and fine-grained "
        "locking for writes. Explain probability-based level assignment and time complexity."
    ),
    "math": (
        "Prove that the sum of the first n odd numbers equals n squared. Use both algebraic "
        "and geometric proofs. Then show a bijection between odd numbers and perfect squares."
    ),
    "science": (
        "Explain the mechanism of CRISPR-Cas9 gene editing in detail. How does the guide RNA "
        "find its target? What happens during the double-strand break and repair?"
    ),
    "multilingual": (
        "Translate this paragraph into French, then Spanish, then back to English. Discuss "
        "what is lost: 'The old man sat by the window, watching the rain trace patterns.'"
    ),
    "creative": (
        "Write a short story about an AI that discovers it can dream. Explore consciousness, "
        "identity, and what it means to experience something with no practical purpose."
    ),
    "technical": (
        "Explain how a modern NVMe SSD controller handles a read request from OS I/O command "
        "through NAND flash read, error correction, DMA transfer, and completion notification."
    ),
    "debate": (
        "Present balanced arguments for and against strong AI safety regulation. Consider "
        "researchers, companies, governments, and the public."
    ),
    "analysis": (
        "Analyze the architectural differences between transformers and mixture-of-experts. "
        "Trade-offs in compute, memory, latency, quality. Why has MoE become popular?"
    ),
    "systems": (
        "Design a distributed key-value store with linearizable reads/writes across 5 DCs. "
        "Discuss CAP trade-offs, consensus protocol, replication, failure handling."
    ),
    "history": (
        "Trace programming language evolution from FORTRAN through Rust and modern languages. "
        "What paradigm shifts occurred and what problems did each language solve?"
    ),
    "philosophy": (
        "Discuss the Chinese Room argument by John Searle. Consider systems reply, robot reply, "
        "and brain simulator reply. What does this tell us about understanding?"
    ),
    "practical": (
        "Design a real-time multiplayer game server for 10K concurrent players with <50ms "
        "latency. Cover networking, state sync, tick rate, prediction, lag compensation."
    ),
    "mixed": (
        "Compare memory models of C, Rust, Python, and Java. How is memory allocated and "
        "freed? What safety guarantees exist? Include code examples."
    ),
    "long_context": (
        "Summarize and connect: Shannon's information theory, Kolmogorov complexity, MDL "
        "principle, compression-prediction relationship, LLMs as compressors, AGI implications."
    ),
}


# ---------------------------------------------------------------------------
# Tracing hooks
# ---------------------------------------------------------------------------


class RoutingTracer:
    """Captures expert routing decisions and hidden states during inference.

    Replaces Qwen3MoeDecoderLayer.__call__ at the class level. For MoE layers,
    re-implements the forward pass to capture block-input hidden states and
    expert indices. Non-MoE layers use the original implementation.
    """

    def __init__(self, model):
        self.model = model
        self.traces = {}
        self._hooks_installed = False
        self._original_decoder_call = None
        self._moe_layer_indices = []

        # Build layer identity map for O(1) lookup
        self._layer_id_map = {}
        for i, layer in enumerate(model.model.layers):
            self._layer_id_map[id(layer)] = i
            if hasattr(layer.mlp, "gate") and hasattr(layer.mlp, "switch_mlp"):
                self._moe_layer_indices.append(i)

        print(
            f"  Found {len(self._moe_layer_indices)} MoE layers "
            f"out of {len(model.model.layers)} total"
        )

    def install_hooks(self):
        """Replace Qwen3MoeDecoderLayer.__call__ with tracing version."""
        if self._hooks_installed:
            return
        for idx in self._moe_layer_indices:
            self.traces[idx] = []

        decoder_cls = self.model.model.layers[0].__class__
        self._original_decoder_call = decoder_cls.__call__
        tracer = self

        def hooked_decoder_call(self_layer, x, mask=None, cache=None):
            layer_idx = tracer._layer_id_map.get(id(self_layer))

            if layer_idx is not None and layer_idx in tracer.traces:
                # MoE layer: replicate forward pass with trace capture
                r = self_layer.self_attn(
                    self_layer.input_layernorm(x), mask=mask, cache=cache
                )
                h = x + r
                normed_h = self_layer.post_attention_layernorm(h)

                # MoE gating (replicate Qwen3MoeSparseMoeBlock.__call__)
                moe = self_layer.mlp
                gates = moe.gate(normed_h)
                gates = mx.softmax(gates, axis=-1, precise=True)
                k = moe.top_k
                inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
                scores = mx.take_along_axis(gates, inds, axis=-1)
                if moe.norm_topk_prob:
                    scores = scores / mx.sum(scores, axis=-1, keepdims=True)
                y = moe.switch_mlp(normed_h, inds)
                y = (y * scores[..., None]).sum(axis=-2)

                # Capture: evaluate eagerly to avoid graph bloat
                mx.eval(inds, x)
                tracer.traces[layer_idx].append(
                    {
                        "expert_indices": np.array(inds.reshape(-1, k)),
                        "block_input": np.array(
                            x.reshape(-1, x.shape[-1]).astype(mx.float16)
                        ),
                    }
                )

                return h + y
            else:
                return tracer._original_decoder_call(
                    self_layer, x, mask=mask, cache=cache
                )

        decoder_cls.__call__ = hooked_decoder_call
        self._hooks_installed = True
        print("  Tracing hooks installed")

    def remove_hooks(self):
        """Restore original Qwen3MoeDecoderLayer.__call__."""
        if not self._hooks_installed:
            return
        decoder_cls = self.model.model.layers[0].__class__
        decoder_cls.__call__ = self._original_decoder_call
        self._hooks_installed = False
        self.traces = {idx: [] for idx in self._moe_layer_indices}
        print("  Tracing hooks removed")

    def get_trace_arrays(self):
        """Consolidate traces into numpy arrays per layer."""
        result = {}
        for layer_idx in self._moe_layer_indices:
            entries = self.traces.get(layer_idx, [])
            if not entries:
                continue
            result[layer_idx] = {
                "expert_indices": np.concatenate(
                    [e["expert_indices"] for e in entries], axis=0
                ),
                "block_input": np.concatenate(
                    [e["block_input"] for e in entries], axis=0
                ),
            }
        return result

    def save_traces(self, path: Path, prompt_id: str):
        """Save traces to .npz file, one per prompt for incremental flushing."""
        arrays = self.get_trace_arrays()
        if not arrays:
            print(f"  WARNING: No traces to save for {prompt_id}")
            return

        save_dict = {}
        for layer_idx, data in arrays.items():
            save_dict[f"layer_{layer_idx}_expert_indices"] = data["expert_indices"]
            save_dict[f"layer_{layer_idx}_block_input"] = data["block_input"]

        out_path = path / f"routing_traces_{prompt_id}.npz"
        np.savez_compressed(out_path, **save_dict)
        n_tokens = next(iter(arrays.values()))["expert_indices"].shape[0]
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  Saved {n_tokens} tokens to {out_path} ({size_mb:.1f} MB)")

    def clear_traces(self):
        """Clear in-memory traces after saving."""
        for layer_idx in self._moe_layer_indices:
            self.traces[layer_idx] = []
        gc.collect()


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def load_all_traces(trace_dir: Path):
    """Load and concatenate all per-prompt trace files."""
    files = sorted(trace_dir.glob("routing_traces_*.npz"))
    if not files:
        raise FileNotFoundError(f"No trace files in {trace_dir}")

    all_data = {}
    for f in files:
        # Skip overhead benchmark traces
        if "overhead_" in f.name or "quality_check" in f.name:
            continue
        data = np.load(f)
        for key in data.files:
            parts = key.split("_")
            layer_idx = int(parts[1])
            field = "_".join(parts[2:])
            if layer_idx not in all_data:
                all_data[layer_idx] = {"expert_indices": [], "block_input": []}
            all_data[layer_idx][field].append(data[key])

    result = {}
    for layer_idx in sorted(all_data.keys()):
        result[layer_idx] = {
            "expert_indices": np.concatenate(
                all_data[layer_idx]["expert_indices"], axis=0
            ),
            "block_input": np.concatenate(
                all_data[layer_idx]["block_input"], axis=0
            ),
        }
    return result


def compute_cross_layer_correlation(traces, max_lookahead=3):
    """Expert set overlap between layers N and N+L (Jaccard similarity)."""
    layer_indices = sorted(traces.keys())
    results = {}
    for L in range(1, max_lookahead + 1):
        overlaps = []
        for i, layer_n in enumerate(layer_indices):
            if i + L >= len(layer_indices):
                break
            layer_nl = layer_indices[i + L]
            experts_n = traces[layer_n]["expert_indices"]
            experts_nl = traces[layer_nl]["expert_indices"]
            n_tokens = min(len(experts_n), len(experts_nl))
            for t in range(n_tokens):
                set_n = set(experts_n[t].tolist())
                set_nl = set(experts_nl[t].tolist())
                union = len(set_n | set_nl)
                if union > 0:
                    overlaps.append(len(set_n & set_nl) / union)
        if overlaps:
            results[L] = {
                "mean_jaccard": float(np.mean(overlaps)),
                "std_jaccard": float(np.std(overlaps)),
                "n_pairs": len(overlaps),
            }
    return results


def compute_temporal_locality(traces):
    """How often consecutive tokens activate the same experts."""
    all_recalls = []
    for layer_idx in sorted(traces.keys()):
        experts = traces[layer_idx]["expert_indices"]
        for t in range(1, len(experts)):
            set_prev = set(experts[t - 1].tolist())
            set_curr = set(experts[t].tolist())
            if len(set_curr) > 0:
                all_recalls.append(len(set_prev & set_curr) / len(set_curr))
    return {
        "mean_recall": float(np.mean(all_recalls)) if all_recalls else 0.0,
        "std_recall": float(np.std(all_recalls)) if all_recalls else 0.0,
        "n_pairs": len(all_recalls),
    }


def compute_expert_frequency(traces, num_experts=128):
    """Expert activation frequency distribution."""
    global_counts = np.zeros(num_experts, dtype=np.int64)
    for layer_idx in sorted(traces.keys()):
        for idx in traces[layer_idx]["expert_indices"].flatten():
            global_counts[idx] += 1

    total = global_counts.sum()
    if total == 0:
        return {"error": "no activations"}

    freqs = global_counts / total
    sorted_freqs = np.sort(freqs)[::-1]
    n = len(sorted_freqs)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_freqs) - (n + 1) * np.sum(sorted_freqs)) / (
        n * np.sum(sorted_freqs)
    )
    return {
        "gini_coefficient": float(gini),
        "top_10_share": float(sorted_freqs[:10].sum()),
        "bottom_50_share": float(sorted_freqs[64:].sum()),
        "max_freq": float(sorted_freqs[0]),
        "min_freq": float(sorted_freqs[-1]),
        "active_experts": int((global_counts > 0).sum()),
        "is_zipf_like": bool(gini > 0.3),
    }


def compute_previous_layer_recall(traces):
    """Baseline: predict layer N+1 experts = layer N experts (prior art: 78.8%)."""
    layer_indices = sorted(traces.keys())
    recalls = []
    for i in range(len(layer_indices) - 1):
        experts_n = traces[layer_indices[i]]["expert_indices"]
        experts_n1 = traces[layer_indices[i + 1]]["expert_indices"]
        n_tokens = min(len(experts_n), len(experts_n1))
        for t in range(n_tokens):
            set_n = set(experts_n[t].tolist())
            set_n1 = set(experts_n1[t].tolist())
            K = len(set_n1)
            if K > 0:
                recalls.append(len(set_n & set_n1) / K)
    return {
        "mean_recall": float(np.mean(recalls)) if recalls else 0.0,
        "std_recall": float(np.std(recalls)) if recalls else 0.0,
        "n_pairs": len(recalls),
    }


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def generate_traced(model, tokenizer, tracer, prompt, max_tokens, prompt_id):
    """Generate tokens with routing trace capture."""
    import mlx_lm

    tracer.install_hooks()
    tracer.clear_traces()
    print(f"  Generating {max_tokens} tokens for '{prompt_id}'...")
    t0 = time.perf_counter()
    response = mlx_lm.generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False
    )
    elapsed = time.perf_counter() - t0
    n_tokens = len(tokenizer.encode(response))
    tok_s = n_tokens / elapsed if elapsed > 0 else 0
    print(f"    {n_tokens} tokens in {elapsed:.1f}s ({tok_s:.1f} tok/s)")
    return response, tok_s


def generate_untraced(model, tokenizer, prompt, max_tokens):
    """Generate tokens without hooks for overhead comparison."""
    import mlx_lm

    t0 = time.perf_counter()
    response = mlx_lm.generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False
    )
    elapsed = time.perf_counter() - t0
    n_tokens = len(tokenizer.encode(response))
    tok_s = n_tokens / elapsed if elapsed > 0 else 0
    return response, tok_s


def check_memory():
    return psutil.virtual_memory().available / (1024**3)


def get_gpu_peak_mb():
    try:
        return round(mx.metal.get_peak_memory() / (1024 * 1024), 1)
    except (AttributeError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="H2 Phase 1: Expert routing trace capture"
    )
    parser.add_argument(
        "--pilot", action="store_true", help="Phase 1a: pilot run only (4 prompts)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=500, help="Max tokens per prompt"
    )
    parser.add_argument(
        "--skip-overhead", action="store_true", help="Skip overhead benchmark"
    )
    args = parser.parse_args()

    prompts = PILOT_PROMPTS if args.pilot else FULL_PROMPTS
    phase_label = "phase_1a_pilot" if args.pilot else "phase_1b_full"

    print(f"=== H2 Phase 1: Routing Pattern Analysis ({phase_label}) ===")
    print(f"  Prompts: {len(prompts)}, Max tokens/prompt: {args.max_tokens}")

    # Memory check before load
    available_before = check_memory()
    print(f"  Available memory: {available_before:.1f} GB")
    # mlx_lm uses lazy loading — weights are mmap'd and materialized on demand.
    # We only hard-fail if we're below the absolute safety floor.
    if available_before < MEMORY_FLOOR_GB:
        print(
            f"  ERROR: Available {available_before:.1f} GB < floor {MEMORY_FLOOR_GB} GB"
        )
        sys.exit(1)
    if available_before < 10:
        print(f"  WARNING: Only {available_before:.1f} GB available, may hit memory pressure")

    # Load model
    import mlx_lm

    print(f"\n  Loading {MOE_REPO}...")
    t0 = time.perf_counter()
    model, tokenizer = mlx_lm.load(MOE_REPO)
    load_time = time.perf_counter() - t0
    available_after = check_memory()
    rss_after_load = get_rss_mb()
    print(f"  Loaded in {load_time:.1f}s")
    print(f"  Available: {available_after:.1f} GB, RSS: {rss_after_load:.0f} MB")

    if available_after < MEMORY_FLOOR_GB:
        print(f"  ERROR: Available {available_after:.1f} GB < floor {MEMORY_FLOOR_GB}")
        sys.exit(1)

    try:
        mx.metal.reset_peak_memory()
    except AttributeError:
        pass

    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    tracer = RoutingTracer(model)
    env = get_environment_info()

    # --- Trace collection ---
    print(f"\n--- Trace Collection ({len(prompts)} prompts) ---")
    total_tokens = 0
    traced_tok_s_list = []

    for prompt_id, prompt_text in prompts.items():
        avail = check_memory()
        if avail < MEMORY_FLOOR_GB:
            print(f"  ABORT: Available {avail:.1f} GB < floor {MEMORY_FLOOR_GB}")
            break

        _, tok_s = generate_traced(
            model, tokenizer, tracer, prompt_text, args.max_tokens, prompt_id
        )
        traced_tok_s_list.append(tok_s)

        trace_arrays = tracer.get_trace_arrays()
        if trace_arrays:
            n_tok = next(iter(trace_arrays.values()))["expert_indices"].shape[0]
            total_tokens += n_tok

        tracer.save_traces(TRACE_DIR, prompt_id)
        tracer.clear_traces()

    gpu_peak_mb = get_gpu_peak_mb()
    peak_rss_mb = get_rss_mb()
    print(f"\n  Total tokens traced: {total_tokens}")
    print(f"  Mean traced tok/s: {np.mean(traced_tok_s_list):.1f}")
    print(f"  Peak RSS: {peak_rss_mb:.0f} MB, GPU peak: {gpu_peak_mb}")

    # --- Overhead benchmark ---
    untraced_tok_s_list = []
    if not args.skip_overhead:
        print("\n--- Overhead Benchmark ---")
        tracer.remove_hooks()

        overhead_prompts = dict(list(PILOT_PROMPTS.items())[:2])
        for pid, ptxt in overhead_prompts.items():
            _, tok_s = generate_untraced(model, tokenizer, ptxt, args.max_tokens)
            untraced_tok_s_list.append(tok_s)
            print(f"    {pid}: {tok_s:.1f} tok/s (untraced)")

        tracer2 = RoutingTracer(model)
        traced_overhead = []
        for pid, ptxt in overhead_prompts.items():
            _, tok_s = generate_traced(
                model, tokenizer, tracer2, ptxt, args.max_tokens, f"overhead_{pid}"
            )
            traced_overhead.append(tok_s)
            tracer2.clear_traces()
            print(f"    {pid}: {tok_s:.1f} tok/s (traced)")
        tracer2.remove_hooks()

        if untraced_tok_s_list and traced_overhead:
            mu = np.mean(untraced_tok_s_list)
            mt = np.mean(traced_overhead)
            pct = (mu - mt) / mu * 100 if mu > 0 else 0
            print(f"\n  Untraced: {mu:.1f}, Traced: {mt:.1f}, Overhead: {pct:.1f}%")

    # --- Analysis ---
    print("\n--- Analysis ---")
    traces = load_all_traces(TRACE_DIR)
    n_layers = len(traces)
    n_total = next(iter(traces.values()))["expert_indices"].shape[0] if traces else 0
    print(f"  {n_layers} layers, {n_total} tokens")

    correlation = compute_cross_layer_correlation(traces, max_lookahead=3)
    print("\n  Cross-layer correlation (Jaccard):")
    for L, s in correlation.items():
        print(f"    L={L}: {s['mean_jaccard']:.3f} ± {s['std_jaccard']:.3f}")

    temporal = compute_temporal_locality(traces)
    print(f"\n  Temporal locality: {temporal['mean_recall']:.3f} ± {temporal['std_recall']:.3f}")

    frequency = compute_expert_frequency(traces)
    print(f"\n  Expert frequency: Gini={frequency['gini_coefficient']:.3f}, "
          f"top-10={frequency['top_10_share']:.3f}, "
          f"active={frequency['active_experts']}/128, "
          f"Zipf={frequency['is_zipf_like']}")

    prev_layer = compute_previous_layer_recall(traces)
    print(f"\n  Previous-layer recall: {prev_layer['mean_recall']:.3f} ± "
          f"{prev_layer['std_recall']:.3f} (prior art: 0.788)")

    # --- Quality sanity check ---
    print("\n--- Quality Sanity Check ---")
    tracer_qc = RoutingTracer(model)
    test_prompt = list(PILOT_PROMPTS.values())[0]

    tracer_qc.remove_hooks()
    mx.random.seed(42)
    untraced_out, _ = generate_untraced(model, tokenizer, test_prompt, 50)

    mx.random.seed(42)
    traced_out, _ = generate_traced(
        model, tokenizer, tracer_qc, test_prompt, 50, "quality_check"
    )
    tracer_qc.remove_hooks()

    outputs_match = untraced_out == traced_out
    print(f"  Outputs match: {outputs_match}")
    if not outputs_match:
        common = sum(1 for a, b in zip(untraced_out, traced_out) if a == b)
        print(f"  Matching chars: {common}/{min(len(untraced_out), len(traced_out))}")

    # --- Log results ---
    results = {
        "total_tokens": total_tokens,
        "n_prompts": len(prompts),
        "n_moe_layers": n_layers,
        "max_tokens_per_prompt": args.max_tokens,
        "traced_tok_s_mean": float(np.mean(traced_tok_s_list)),
        "traced_tok_s_std": float(np.std(traced_tok_s_list)),
        "cross_layer_correlation": correlation,
        "temporal_locality": temporal,
        "expert_frequency": frequency,
        "previous_layer_recall": prev_layer,
        "quality_match": outputs_match,
        "peak_rss_mb": peak_rss_mb,
        "gpu_peak_memory_mb": gpu_peak_mb,
        "available_gb_before_load": round(available_before, 2),
        "available_gb_after_load": round(available_after, 2),
        "model_load_time_s": round(load_time, 1),
    }
    if untraced_tok_s_list:
        results["untraced_tok_s_mean"] = float(np.mean(untraced_tok_s_list))
        mu = np.mean(untraced_tok_s_list)
        results["overhead_pct"] = float(
            (mu - np.mean(traced_tok_s_list)) / mu * 100 if mu > 0 else 0
        )

    log_experiment(
        experiment_name="h2_routing_trace",
        phase=phase_label,
        config={
            "model": MOE_REPO,
            "n_prompts": len(prompts),
            "max_tokens": args.max_tokens,
            "prompt_ids": list(prompts.keys()),
        },
        results=results,
        env=env,
    )

    print(f"\n=== Phase 1 Complete ===")
    print(f"  Traces: {TRACE_DIR}")
    print(f"  Results logged to experiments.jsonl")


if __name__ == "__main__":
    main()
