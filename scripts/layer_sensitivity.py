"""
Phase 1: Layer Sensitivity Profiling for Layer LOD experiment.

Measures per-layer quantization sensitivity by computing perplexity impact
when individual layers are quantized to lower precision while keeping all
others at Q4 baseline.

Uses pure MLX (no torch) — loads a pre-quantized Q4 model and measures
perplexity change when re-quantizing individual layers to Q2 or Q3.

Usage:
    uv run python scripts/layer_sensitivity.py [--model MODEL_ID] [--n-samples N]
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.experiment_utils import get_environment_info, log_experiment


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------


def compute_perplexity(model, tokenizer, texts: list[str], max_len: int = 512) -> float:
    """Compute perplexity on a list of text samples using MLX.

    Uses model.__call__ with cache=None to ensure correct forward pass
    (handles position IDs, masks, etc. internally via mlx-lm model code).
    """
    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        tokens = tokens[:max_len]

        input_ids = mx.array(tokens[:-1])[None, :]  # (1, seq_len)
        target_ids = mx.array(tokens[1:])  # (seq_len,)

        # Pass cache=None explicitly to ensure the model handles position
        # encoding and masking correctly (mlx-lm models accept cache kwarg)
        logits = model(input_ids, cache=None)
        logits = logits[0]  # (1, seq_len, vocab_size) → (seq_len, vocab_size)

        # Cross-entropy loss
        log_probs = nn.losses.cross_entropy(logits, target_ids, reduction="sum")
        mx.eval(log_probs)
        total_loss += log_probs.item()
        total_tokens += len(tokens) - 1

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


# ---------------------------------------------------------------------------
# Layer manipulation
# ---------------------------------------------------------------------------


def get_quantized_layer_names(model) -> list[str]:
    """Find all quantized linear layers in the model's transformer blocks."""
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear) and "layers." in name:
            layers.append(name)
    return sorted(layers)


def get_layer_index(name: str) -> int:
    """Extract the transformer block index from a layer name."""
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return -1


def get_layer_type(name: str) -> str:
    """Classify layer as 'attention' or 'mlp' or 'other'."""
    if any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj", "self_attn"]):
        return "attention"
    elif any(k in name for k in ["gate_proj", "up_proj", "down_proj", "mlp"]):
        return "mlp"
    return "other"


def _navigate_to_module(model, path: str):
    """Navigate a dotted path to find a module, handling numeric indices."""
    parts = path.split(".")
    obj = model
    for part in parts:
        try:
            idx = int(part)
            obj = obj[idx]
        except (ValueError, TypeError):
            obj = getattr(obj, part)
    return obj


def _navigate_to_parent(model, path: str):
    """Navigate to the parent of the target, returning (parent, last_part)."""
    parts = path.split(".")
    obj = model
    for part in parts[:-1]:
        try:
            idx = int(part)
            obj = obj[idx]
        except (ValueError, TypeError):
            obj = getattr(obj, part)
    return obj, parts[-1]


def requantize_layer(model, layer_name: str, target_bits: int, group_size: int = 64):
    """Re-quantize a single QuantizedLinear layer to a different bit-width.

    Returns the original weights for manual restoration.
    """
    module = _navigate_to_module(model, layer_name)

    if not isinstance(module, nn.QuantizedLinear):
        return None

    # Save original quantized weights
    original_weight = module.weight
    original_scales = module.scales
    original_biases = getattr(module, "biases", None)
    original_bits = module.bits
    original_group_size = module.group_size

    # Dequantize to get full-precision weights
    # QuantizedLinear stores: weight (packed), scales, biases
    # We need to dequantize, then re-quantize at the target bits
    weight_fp = mx.dequantize(
        module.weight, module.scales, module.biases,
        module.group_size, module.bits,
    )

    # Re-quantize at target bits
    # No mx.eval() here — let MLX batch the computation lazily.
    # The caller should eval after all layers in a block are re-quantized.
    new_weight, new_scales, new_biases = mx.quantize(
        weight_fp, group_size=group_size, bits=target_bits
    )

    # Replace weights in-place
    module.weight = new_weight
    module.scales = new_scales
    module.biases = new_biases
    module.bits = target_bits
    module.group_size = group_size

    # Return restoration info
    return {
        "weight": original_weight,
        "scales": original_scales,
        "biases": original_biases,
        "bits": original_bits,
        "group_size": original_group_size,
    }


def restore_layer(model, layer_name: str, original: dict):
    """Restore a layer to its original quantization state."""
    module = _navigate_to_module(model, layer_name)

    module.weight = original["weight"]
    module.scales = original["scales"]
    if original["biases"] is not None:
        module.biases = original["biases"]
    module.bits = original["bits"]
    module.group_size = original["group_size"]


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------

CALIBRATION_TEXTS = [
    "The transformer architecture has revolutionized natural language processing. "
    "Self-attention mechanisms allow the model to attend to all positions in the input "
    "sequence simultaneously, enabling parallel processing and capturing long-range "
    "dependencies more effectively than recurrent architectures.",

    "In distributed computing systems, consensus protocols ensure that multiple nodes "
    "agree on a single value despite potential failures. The Raft protocol provides "
    "understandable leader election and log replication mechanisms.",

    "Quantization reduces the precision of neural network weights from floating point "
    "to lower bit representations. This compression technique trades a small amount of "
    "accuracy for significant memory and computational savings, enabling deployment on "
    "resource-constrained devices.",

    "The theory of general relativity describes gravity as the curvature of spacetime "
    "caused by mass and energy. This framework unifies space and time into a single "
    "four-dimensional manifold and predicts phenomena such as gravitational waves.",

    "Modern operating systems use virtual memory to provide each process with its own "
    "address space. Page tables map virtual addresses to physical frames, and the MMU "
    "translates addresses in hardware for minimal overhead.",

    "Gradient descent optimizes a loss function by iteratively updating parameters in "
    "the direction of steepest descent. Variants like Adam combine momentum and adaptive "
    "learning rates to achieve faster convergence on non-convex landscapes.",

    "Binary search trees maintain sorted data for efficient lookup, insertion, and deletion. "
    "Self-balancing variants such as red-black trees guarantee O(log n) worst-case performance "
    "by enforcing structural invariants through rotations.",

    "The PageRank algorithm assigns importance scores to web pages based on the link structure "
    "of the World Wide Web. Pages with many inbound links from high-importance pages receive "
    "higher scores, creating a recursive definition of relevance.",
]


def load_calibration_data(tokenizer, n_samples: int = 8) -> list[str]:
    """Load calibration texts for perplexity measurement."""
    return CALIBRATION_TEXTS[:n_samples]


# ---------------------------------------------------------------------------
# Main sensitivity profiling
# ---------------------------------------------------------------------------


def profile_layer_sensitivity(
    model_id: str,
    target_bits_list: list[int] = [2, 3],
    n_samples: int = 8,
    max_len: int = 512,
):
    """Profile per-layer quantization sensitivity.

    For each transformer block layer:
    1. Re-quantize just that layer to target_bits
    2. Measure perplexity on calibration data
    3. Compute perplexity delta vs baseline (all Q4)
    """
    from mlx_lm import load

    print(f"\nLoading model: {model_id}")
    model, tokenizer = load(model_id)
    mx.eval(model.parameters())

    # Get calibration data
    texts = load_calibration_data(tokenizer, n_samples)
    print(f"Calibration: {len(texts)} samples, max_len={max_len}")

    # Measure baseline perplexity (all layers at Q4)
    print("\nMeasuring baseline perplexity (all Q4)...")
    t0 = time.time()
    baseline_ppl = compute_perplexity(model, tokenizer, texts, max_len)
    baseline_time = time.time() - t0
    print(f"  Baseline PPL: {baseline_ppl:.4f} ({baseline_time:.1f}s)")

    # Find all quantized layers
    quant_layers = get_quantized_layer_names(model)
    print(f"\nFound {len(quant_layers)} quantized layers")

    # Group layers by transformer block for block-level analysis
    block_layers = {}
    for name in quant_layers:
        idx = get_layer_index(name)
        if idx >= 0:
            block_layers.setdefault(idx, []).append(name)

    n_blocks = len(block_layers)
    print(f"Transformer blocks: {n_blocks}")

    # Profile each block at each target bit-width
    results = []
    for target_bits in target_bits_list:
        print(f"\n{'='*60}")
        print(f"Profiling at Q{target_bits} substitution")
        print(f"{'='*60}")

        for block_idx in sorted(block_layers.keys()):
            layer_names = block_layers[block_idx]
            print(f"\n  Block {block_idx}/{n_blocks-1} ({len(layer_names)} layers)...")

            # Re-quantize all layers in this block
            originals = {}
            for name in layer_names:
                orig = requantize_layer(model, name, target_bits)
                if orig:
                    originals[name] = orig

            # Measure perplexity
            t0 = time.time()
            ppl = compute_perplexity(model, tokenizer, texts, max_len)
            elapsed = time.time() - t0

            # Compute delta
            ppl_delta = ppl - baseline_ppl
            ppl_ratio = ppl / baseline_ppl if baseline_ppl > 0 else float("inf")

            # Classify layer types in this block
            types = set(get_layer_type(name) for name in layer_names)

            result = {
                "block_index": block_idx,
                "target_bits": target_bits,
                "perplexity": round(ppl, 4),
                "ppl_delta": round(ppl_delta, 4),
                "ppl_ratio": round(ppl_ratio, 4),
                "layer_types": sorted(types),
                "n_layers": len(layer_names),
                "eval_time_s": round(elapsed, 1),
            }
            results.append(result)

            # Visual indicator of sensitivity
            sensitivity = "▓" * min(int(abs(ppl_delta) * 10), 20)
            print(f"    PPL: {ppl:.4f} (Δ{ppl_delta:+.4f}, ×{ppl_ratio:.4f}) "
                  f"{sensitivity}")

            # Restore original weights
            for name, orig in originals.items():
                restore_layer(model, name, orig)

        # Log per-bit-width results
        log_experiment(
            experiment_name=f"layer_sensitivity_{model_id.split('/')[-1]}_q{target_bits}",
            phase="layer_sensitivity",
            config={
                "model_id": model_id,
                "target_bits": target_bits,
                "n_samples": n_samples,
                "max_len": max_len,
                "n_blocks": n_blocks,
            },
            results={
                "baseline_ppl": baseline_ppl,
                "block_results": [r for r in results if r["target_bits"] == target_bits],
            },
        )

    # Analysis: confirm/deny U-shape
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS")
    print("=" * 60)

    for target_bits in target_bits_list:
        bit_results = [r for r in results if r["target_bits"] == target_bits]
        deltas = [r["ppl_delta"] for r in bit_results]
        indices = [r["block_index"] for r in bit_results]

        if not deltas:
            continue

        print(f"\nQ{target_bits} substitution:")

        # Compute sensitivity by position
        n = len(deltas)
        first_20pct = deltas[:max(1, n // 5)]
        middle_60pct = deltas[n // 5: 4 * n // 5]
        last_20pct = deltas[4 * n // 5:]

        avg_first = np.mean(first_20pct) if first_20pct else 0
        avg_middle = np.mean(middle_60pct) if middle_60pct else 0
        avg_last = np.mean(last_20pct) if last_20pct else 0

        print(f"  First 20% avg Δ:  {avg_first:+.4f}")
        print(f"  Middle 60% avg Δ: {avg_middle:+.4f}")
        print(f"  Last 20% avg Δ:   {avg_last:+.4f}")

        # U-shape test: edges should be >2x middle
        edge_avg = (avg_first + avg_last) / 2
        if avg_middle > 0:
            ratio = edge_avg / avg_middle
            u_shape = ratio > 2.0
            print(f"  Edge/middle ratio: {ratio:.2f}x — "
                  f"{'✓ U-shape CONFIRMED' if u_shape else '✗ U-shape NOT confirmed'}")
        else:
            print(f"  Middle Δ ≤ 0 — cannot compute ratio")

        # Most/least sensitive blocks
        sorted_by_delta = sorted(bit_results, key=lambda r: r["ppl_delta"], reverse=True)
        print(f"\n  Most sensitive blocks (highest Δ):")
        for r in sorted_by_delta[:5]:
            print(f"    Block {r['block_index']:3d}: Δ{r['ppl_delta']:+.4f}")
        print(f"  Least sensitive blocks (lowest Δ):")
        for r in sorted_by_delta[-5:]:
            print(f"    Block {r['block_index']:3d}: Δ{r['ppl_delta']:+.4f}")

    # Clean up
    del model, tokenizer

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Layer LOD Phase 1: Sensitivity Profiling")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit",
                        help="Model ID to profile")
    parser.add_argument("--n-samples", type=int, default=8,
                        help="Number of calibration samples")
    parser.add_argument("--max-len", type=int, default=512,
                        help="Maximum sequence length for perplexity")
    parser.add_argument("--bits", nargs="+", type=int, default=[2, 3],
                        help="Target bit-widths to test")
    args = parser.parse_args()

    print("=" * 60)
    print("Layer LOD Experiment — Phase 1: Sensitivity Profiling")
    print("=" * 60)

    env = get_environment_info()
    print(f"Environment: {env['chip']}, {env['memory_gb']} GB RAM")

    results = profile_layer_sensitivity(
        model_id=args.model,
        target_bits_list=args.bits,
        n_samples=args.n_samples,
        max_len=args.max_len,
    )

    print(f"\nDone. {len(results)} measurements logged to experiments.jsonl")


if __name__ == "__main__":
    main()
