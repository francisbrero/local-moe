"""
routing_predictor.py — Phase 2: Expert routing predictor training for H2.

Trains per-layer linear predictors on pre-attention hidden states to predict
expert routing decisions at lookahead L=1,2,3.

Uses traces captured by routing_trace.py (Phase 1).

Usage:
    uv run python scripts/routing_predictor.py [--trace-dir routing_traces]
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np

from experiment_utils import get_environment_info, get_rss_mb, log_experiment

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRACE_DIR = Path(__file__).parent.parent / "routing_traces"
NUM_EXPERTS = 128
TOP_K = 8
HIDDEN_DIM = 2048
PREDICTOR_HIDDEN = 512  # intermediate dim for 2-layer predictor
EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 64


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_traces(trace_dir: Path):
    """Load all trace files, return dict: layer_idx -> {expert_indices, block_input}."""
    files = sorted(trace_dir.glob("routing_traces_*.npz"))
    files = [f for f in files if "overhead_" not in f.name and "quality_check" not in f.name]
    if not files:
        raise FileNotFoundError(f"No trace files in {trace_dir}")

    all_data = {}
    prompt_boundaries = {}  # layer_idx -> list of (start, end) tuples per prompt

    offset = 0
    for f in files:
        data = np.load(f)
        prompt_id = f.stem.replace("routing_traces_", "")
        n_tokens = None
        file_layers = set()  # Track layers in this file only

        for key in data.files:
            # Keys are "layer_{idx}_{field}" — split at first 2 underscores
            parts = key.split("_", 2)  # ['layer', '12', 'expert_indices']
            layer_idx = int(parts[1])
            field = parts[2]
            if layer_idx not in all_data:
                all_data[layer_idx] = {"expert_indices": [], "block_input": []}
            arr = data[key]
            all_data[layer_idx][field].append(arr)
            file_layers.add(layer_idx)
            if n_tokens is None:
                n_tokens = arr.shape[0]
            else:
                # Validate all layers in this file have same token count
                assert arr.shape[0] == n_tokens, (
                    f"Token count mismatch in {f}: {arr.shape[0]} vs {n_tokens}"
                )

        if n_tokens:
            # Only update boundaries for layers present in THIS file
            for layer_idx in file_layers:
                if layer_idx not in prompt_boundaries:
                    prompt_boundaries[layer_idx] = []
                prompt_boundaries[layer_idx].append(
                    (offset, offset + n_tokens, prompt_id)
                )
            offset += n_tokens

    result = {}
    for layer_idx in sorted(all_data.keys()):
        result[layer_idx] = {
            "expert_indices": np.concatenate(all_data[layer_idx]["expert_indices"], axis=0),
            "block_input": np.concatenate(all_data[layer_idx]["block_input"], axis=0),
        }

    return result, prompt_boundaries


def split_by_prompt(traces, prompt_boundaries, train_frac=0.7, val_frac=0.15):
    """Split data by prompt into train/val/test sets."""
    # Get unique prompts from the first layer
    first_layer = sorted(prompt_boundaries.keys())[0]
    prompts = prompt_boundaries[first_layer]
    n_prompts = len(prompts)

    np.random.seed(42)
    perm = np.random.permutation(n_prompts)

    n_train = max(1, int(n_prompts * train_frac))
    n_val = max(1, int(n_prompts * val_frac))

    train_idx = set(perm[:n_train].tolist())
    val_idx = set(perm[n_train : n_train + n_val].tolist())
    test_idx = set(perm[n_train + n_val :].tolist())

    # If test set is empty (few prompts), steal from train
    if not test_idx:
        test_idx = {perm[0]}
        train_idx.discard(perm[0])

    def gather_tokens(layer_idx, indices):
        bounds = prompt_boundaries[layer_idx]
        token_indices = []
        for i in indices:
            start, end, _ = bounds[i]
            token_indices.extend(range(start, end))
        return np.array(token_indices, dtype=np.int64)

    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "gather": gather_tokens,
        "prompts": prompts,
    }


# ---------------------------------------------------------------------------
# Predictor model
# ---------------------------------------------------------------------------


class ExpertPredictor(nn.Module):
    """2-layer linear predictor: hidden_dim -> predictor_hidden -> num_experts."""

    def __init__(self, input_dim=HIDDEN_DIM, hidden_dim=PREDICTOR_HIDDEN,
                 output_dim=NUM_EXPERTS):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def __call__(self, x):
        x = nn.relu(self.fc1(x))
        return self.fc2(x)  # raw logits, sigmoid applied at loss


def multi_label_bce_loss(logits, targets):
    """Binary cross-entropy for multi-hot targets."""
    # Numerically stable BCE: -t*log(sigmoid(x)) - (1-t)*log(1-sigmoid(x))
    # = max(x,0) - x*t + log(1 + exp(-|x|))
    pos = mx.maximum(logits, 0)
    neg_abs = mx.abs(logits)
    loss = pos - logits * targets + mx.log(1 + mx.exp(-neg_abs))
    return loss.mean()


def expert_indices_to_multihot(indices, num_experts=NUM_EXPERTS):
    """Convert [T, K] expert indices to [T, num_experts] multi-hot vectors."""
    T, K = indices.shape
    multihot = np.zeros((T, num_experts), dtype=np.float32)
    rows = np.repeat(np.arange(T), K)
    cols = indices.flatten()
    multihot[rows, cols] = 1.0
    return multihot


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_predictor(X_train, Y_train, X_val, Y_val, epochs=EPOCHS, lr=LR):
    """Train a single ExpertPredictor on the given data."""
    model = ExpertPredictor()
    mx.eval(model.parameters())
    optimizer = optim.Adam(learning_rate=lr)

    loss_and_grad = nn.value_and_grad(model, lambda m, x, y: multi_label_bce_loss(m(x), y))

    best_val_loss = float("inf")
    best_weights = None
    patience = 10
    no_improve = 0

    n_train = X_train.shape[0]
    n_batches = max(1, n_train // BATCH_SIZE)

    for epoch in range(epochs):
        perm = mx.array(np.random.permutation(n_train))
        epoch_loss = 0.0

        for b in range(n_batches):
            start = b * BATCH_SIZE
            end = min(start + BATCH_SIZE, n_train)
            idx = perm[start:end]
            xb = X_train[idx]
            yb = Y_train[idx]

            loss, grads = loss_and_grad(model, xb, yb)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            epoch_loss += loss.item()

        epoch_loss /= n_batches

        val_logits = model(X_val)
        val_loss = multi_label_bce_loss(val_logits, Y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save a deep copy of weights as flat list of (name, array) tuples
            best_weights = [(k, mx.array(v)) for k, v in tree_flatten(model.parameters())]
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_weights is not None:
        model.load_weights(best_weights)

    return model, best_val_loss


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


def evaluate_predictor(model, X_test, Y_test_indices, k=TOP_K):
    """Evaluate prediction metrics on test set.

    Args:
        model: trained ExpertPredictor
        X_test: [T, D] hidden states
        Y_test_indices: [T, K] actual expert indices
        k: number of experts to predict
    """
    logits = model(X_test)
    mx.eval(logits)
    logits_np = np.array(logits)

    T = logits_np.shape[0]
    predicted_indices = np.argpartition(logits_np, -k, axis=-1)[:, -k:]

    recalls = []
    precisions = []
    jaccards = []
    exact_matches = []

    for t in range(T):
        pred_set = set(predicted_indices[t].tolist())
        actual_set = set(Y_test_indices[t].tolist())

        intersection = len(pred_set & actual_set)
        union = len(pred_set | actual_set)

        recall = intersection / len(actual_set) if len(actual_set) > 0 else 0
        precision = intersection / len(pred_set) if len(pred_set) > 0 else 0
        jaccard = intersection / union if union > 0 else 0
        exact = 1.0 if pred_set == actual_set else 0.0

        recalls.append(recall)
        precisions.append(precision)
        jaccards.append(jaccard)
        exact_matches.append(exact)

    return {
        "recall_at_k": {"mean": float(np.mean(recalls)), "std": float(np.std(recalls))},
        "precision_at_k": {"mean": float(np.mean(precisions)), "std": float(np.std(precisions))},
        "jaccard_overlap": {"mean": float(np.mean(jaccards)), "std": float(np.std(jaccards))},
        "exact_set_match": {"mean": float(np.mean(exact_matches)), "std": float(np.std(exact_matches))},
        "n_test_tokens": T,
    }


def measure_predictor_latency(model, X_sample, n_runs=100):
    """Measure predictor inference latency per token."""
    # Warm up
    _ = model(X_sample[:1])
    mx.eval(_)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        logits = model(X_sample[:1])
        mx.eval(logits)
        times.append(time.perf_counter() - t0)

    return {
        "latency_ms_mean": float(np.mean(times) * 1000),
        "latency_ms_std": float(np.std(times) * 1000),
        "latency_ms_p50": float(np.percentile(times, 50) * 1000),
        "latency_ms_p95": float(np.percentile(times, 95) * 1000),
    }


# ---------------------------------------------------------------------------
# Co-activation buddy table
# ---------------------------------------------------------------------------


def build_buddy_table(traces, top_n=5):
    """Build per-layer co-activation lookup: for each expert, find top-N buddies.

    Buddies are experts that most frequently co-activate with a given expert
    *within the same layer*. Expert IDs are only meaningful within a single
    layer, so we keep separate tables per layer.
    Used for BuddyMoE-style substitution on prediction misses.
    """
    buddy_table = {}

    for layer_idx in sorted(traces.keys()):
        coactivation = np.zeros((NUM_EXPERTS, NUM_EXPERTS), dtype=np.int64)
        indices = traces[layer_idx]["expert_indices"]  # [T, K]
        for t in range(len(indices)):
            experts = indices[t].tolist()
            for i, e1 in enumerate(experts):
                for e2 in experts[i + 1 :]:
                    coactivation[e1, e2] += 1
                    coactivation[e2, e1] += 1

        # For each expert, find top-N co-activating buddies in this layer
        layer_table = {}
        for e in range(NUM_EXPERTS):
            counts = coactivation[e].copy()
            counts[e] = 0  # exclude self
            top_buddies = np.argsort(counts)[-top_n:][::-1]
            layer_table[e] = {
                "buddies": top_buddies.tolist(),
                "counts": counts[top_buddies].tolist(),
            }
        buddy_table[layer_idx] = layer_table

    return buddy_table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="H2 Phase 2: Expert routing predictor training"
    )
    parser.add_argument(
        "--trace-dir", type=str, default=str(TRACE_DIR), help="Trace directory"
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    print("=== H2 Phase 2: Predictor Design + Training ===")

    # Load traces
    print("\n--- Loading traces ---")
    traces, prompt_boundaries = load_traces(trace_dir)
    layer_indices = sorted(traces.keys())
    n_layers = len(layer_indices)
    n_tokens = traces[layer_indices[0]]["expert_indices"].shape[0]
    print(f"  {n_layers} layers, {n_tokens} tokens")

    # Split by prompt
    split = split_by_prompt(traces, prompt_boundaries)
    print(f"  Train prompts: {len(split['train_idx'])}, "
          f"Val: {len(split['val_idx'])}, Test: {len(split['test_idx'])}")

    env = get_environment_info()
    try:
        mx.metal.reset_peak_memory()
    except AttributeError:
        try:
            mx.reset_peak_memory()
        except AttributeError:
            pass

    # Train predictors for each lookahead
    all_results = {}
    for L in [1, 2, 3]:
        print(f"\n--- Lookahead L={L} ---")
        layer_metrics = []

        for i, layer_n in enumerate(layer_indices):
            if i + L >= len(layer_indices):
                break
            layer_target = layer_indices[i + L]

            # Gather data
            train_tokens = split["gather"](layer_n, split["train_idx"])
            val_tokens = split["gather"](layer_n, split["val_idx"])
            test_tokens = split["gather"](layer_n, split["test_idx"])

            if len(train_tokens) < 10 or len(test_tokens) < 5:
                continue

            # Input: block-input hidden states at layer N
            X_train = mx.array(traces[layer_n]["block_input"][train_tokens])
            X_val = mx.array(traces[layer_n]["block_input"][val_tokens])
            X_test = mx.array(traces[layer_n]["block_input"][test_tokens])

            # Target: expert indices at layer N+L (multi-hot)
            Y_train_mh = mx.array(
                expert_indices_to_multihot(traces[layer_target]["expert_indices"][train_tokens])
            )
            Y_val_mh = mx.array(
                expert_indices_to_multihot(traces[layer_target]["expert_indices"][val_tokens])
            )
            Y_test_indices = traces[layer_target]["expert_indices"][test_tokens]

            # Train
            model, val_loss = train_predictor(
                X_train, Y_train_mh, X_val, Y_val_mh, epochs=args.epochs, lr=args.lr
            )

            # Evaluate
            metrics = evaluate_predictor(model, X_test, Y_test_indices)
            metrics["layer_n"] = layer_n
            metrics["layer_target"] = layer_target
            metrics["val_loss"] = val_loss
            layer_metrics.append(metrics)

            if i % 10 == 0:
                print(f"    Layer {layer_n}→{layer_target}: "
                      f"recall@{TOP_K}={metrics['recall_at_k']['mean']:.3f}, "
                      f"jaccard={metrics['jaccard_overlap']['mean']:.3f}")

        if layer_metrics:
            mean_recall = np.mean([m["recall_at_k"]["mean"] for m in layer_metrics])
            mean_precision = np.mean([m["precision_at_k"]["mean"] for m in layer_metrics])
            mean_jaccard = np.mean([m["jaccard_overlap"]["mean"] for m in layer_metrics])
            mean_exact = np.mean([m["exact_set_match"]["mean"] for m in layer_metrics])

            all_results[f"L{L}"] = {
                "mean_recall_at_k": float(mean_recall),
                "mean_precision_at_k": float(mean_precision),
                "mean_jaccard_overlap": float(mean_jaccard),
                "mean_exact_set_match": float(mean_exact),
                "n_layer_pairs": len(layer_metrics),
                "per_layer": layer_metrics,
            }

            print(f"\n  L={L} aggregate: recall@{TOP_K}={mean_recall:.3f}, "
                  f"precision={mean_precision:.3f}, jaccard={mean_jaccard:.3f}, "
                  f"exact={mean_exact:.3f}")

    # Predictor latency measurement
    print("\n--- Predictor Latency ---")
    sample_model = ExpertPredictor()
    mx.eval(sample_model.parameters())
    sample_x = mx.array(traces[layer_indices[0]]["block_input"][:10])
    latency = measure_predictor_latency(sample_model, sample_x)
    print(f"  Per-token: {latency['latency_ms_mean']:.3f}ms ± {latency['latency_ms_std']:.3f}ms")
    print(f"  Per-token (48 layers): {latency['latency_ms_mean'] * 48:.1f}ms")

    # Buddy table
    print("\n--- Co-activation Buddy Table ---")
    buddy_table = build_buddy_table(traces)
    n_layers_with_buddies = len(buddy_table)
    print(f"  Built per-layer buddy tables for {n_layers_with_buddies} layers")
    # Save buddy table (convert int keys to strings for JSON)
    buddy_path = trace_dir / "buddy_table.json"
    serializable = {
        str(layer): {str(e): v for e, v in experts.items()}
        for layer, experts in buddy_table.items()
    }
    with open(buddy_path, "w") as f:
        json.dump(serializable, f)
    print(f"  Saved to {buddy_path}")

    # GPU peak memory
    try:
        gpu_peak = round(mx.get_peak_memory() / (1024 * 1024), 1)
    except AttributeError:
        try:
            gpu_peak = round(mx.metal.get_peak_memory() / (1024 * 1024), 1)
        except (AttributeError, RuntimeError):
            gpu_peak = None

    peak_rss = get_rss_mb()

    # Success gate check
    L2_results = all_results.get("L2", {})
    L2_recall = L2_results.get("mean_recall_at_k", 0)
    gate_passed = L2_recall >= 0.90
    print(f"\n--- Success Gate ---")
    print(f"  L=2 recall@{TOP_K}: {L2_recall:.3f} (gate: ≥0.90)")
    print(f"  Gate: {'PASSED' if gate_passed else 'FAILED'}")

    # Log results
    results = {
        "lookahead_results": all_results,
        "predictor_latency": latency,
        "peak_rss_mb": peak_rss,
        "gpu_peak_memory_mb": gpu_peak,
        "gate_passed": gate_passed,
        "n_tokens_total": n_tokens,
        "n_layers": n_layers,
    }

    log_experiment(
        experiment_name="h2_routing_predictor",
        phase="phase_2_predictor",
        config={
            "hidden_dim": HIDDEN_DIM,
            "predictor_hidden": PREDICTOR_HIDDEN,
            "num_experts": NUM_EXPERTS,
            "top_k": TOP_K,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": BATCH_SIZE,
        },
        results=results,
        env=env,
    )

    print(f"\n=== Phase 2 Complete ===")
    print(f"  Results logged to experiments.jsonl")


if __name__ == "__main__":
    main()
