"""
checkpoint_audit.py — Phase 0: Inspect MoE checkpoint layouts.

Downloads metadata only (config.json, model index) for target MoE models and
measures non-expert vs expert weight sizes, shard layout, tensor contiguity.

This is the ONLY script in the experiment that performs network I/O (metadata
download). All subsequent phases work offline.

Usage:
    uv run python scripts/checkpoint_audit.py
"""

import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, model_info

from experiment_utils import get_environment_info, log_experiment

# ---------------------------------------------------------------------------
# Target MoE models to audit
# ---------------------------------------------------------------------------

MOE_MODELS = [
    {
        "repo": "Qwen/Qwen3-30B-A3B",
        "description": "Qwen3 30B MoE, 3B active params",
        "preferred": True,
    },
    {
        "repo": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "description": "Mixtral 8x7B, 12.9B active params",
        "preferred": False,
    },
]

# Also check quantized MLX variants
MLX_MOE_MODELS = [
    {
        "repo": "mlx-community/Qwen3-30B-A3B-4bit",
        "description": "Qwen3 30B MoE 4-bit quantized for MLX",
        "preferred": True,
    },
    {
        "repo": "mlx-community/Qwen3-30B-A3B-3bit",
        "description": "Qwen3 30B MoE 3-bit quantized for MLX",
        "preferred": True,
    },
    {
        "repo": "mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit",
        "description": "Mixtral 8x7B 4-bit quantized for MLX",
        "preferred": False,
    },
]


# ---------------------------------------------------------------------------
# Audit functions
# ---------------------------------------------------------------------------


def fetch_config(repo: str) -> dict | None:
    """Download and parse config.json from HF hub."""
    try:
        path = hf_hub_download(repo, "config.json")
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  WARNING: Cannot fetch config for {repo}: {e}")
        return None


def fetch_model_index(repo: str) -> dict | None:
    """Download and parse model.safetensors.index.json if it exists."""
    try:
        path = hf_hub_download(repo, "model.safetensors.index.json")
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def extract_moe_config(config: dict) -> dict:
    """Extract MoE-specific parameters from config.json."""
    # Qwen3 MoE uses these fields
    num_experts = config.get("num_experts", config.get("num_local_experts", 0))
    num_experts_per_tok = config.get(
        "num_experts_per_tok", config.get("num_experts_per_tok", 0)
    )
    # Mixtral uses router-specific config
    if num_experts == 0 and "router" in str(config):
        num_experts = config.get("num_local_experts", 8)
        num_experts_per_tok = config.get("num_experts_per_tok", 2)

    hidden_size = config.get("hidden_size", 0)
    intermediate_size = config.get("intermediate_size", 0)
    num_hidden_layers = config.get("num_hidden_layers", 0)
    num_kv_heads = config.get("num_key_value_heads", config.get("num_attention_heads", 0))
    head_dim = config.get("head_dim", 0)
    if head_dim == 0 and hidden_size > 0:
        n_heads = config.get("num_attention_heads", 1)
        head_dim = hidden_size // n_heads

    # MoE intermediate size may differ from dense intermediate
    moe_intermediate_size = config.get("moe_intermediate_size", intermediate_size)

    return {
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "moe_intermediate_size": moe_intermediate_size,
        "num_hidden_layers": num_hidden_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "vocab_size": config.get("vocab_size", 0),
    }


def estimate_expert_sizes(moe_config: dict) -> dict:
    """Estimate per-expert and total expert weight sizes at different quantizations.

    Each MoE expert typically has 3 weight matrices for the FFN:
    - gate_proj: hidden_size x moe_intermediate_size
    - up_proj: hidden_size x moe_intermediate_size
    - down_proj: moe_intermediate_size x hidden_size
    """
    h = moe_config["hidden_size"]
    ffn = moe_config["moe_intermediate_size"]
    n_experts = moe_config["num_experts"]
    n_layers = moe_config["num_hidden_layers"]

    if h == 0 or ffn == 0 or n_experts == 0:
        return {"error": "Missing dimensions for expert size calculation"}

    # Parameters per expert FFN: 3 * h * ffn (gate + up + down)
    params_per_expert = 3 * h * ffn

    # Size at different quantizations
    sizes = {}
    for bits, label in [(16, "fp16"), (4, "4bit"), (3, "3bit"), (2, "2bit")]:
        bytes_per_expert = params_per_expert * bits / 8
        expert_mb = bytes_per_expert / (1024**2)
        total_expert_gb = (bytes_per_expert * n_experts * n_layers) / (1024**3)
        sizes[label] = {
            "bytes_per_expert": int(bytes_per_expert),
            "mb_per_expert": round(expert_mb, 2),
            "total_expert_gb": round(total_expert_gb, 2),
            "total_experts": n_experts * n_layers,
        }

    return sizes


def estimate_non_expert_size(moe_config: dict) -> dict:
    """Estimate non-expert weight sizes (attention, embeddings, routing, norms).

    Non-expert weights include:
    - Embeddings: vocab_size * hidden_size
    - Per-layer attention: Q, K, V, O projections
    - Per-layer norms: RMSNorm weights
    - Per-layer router: hidden_size * num_experts (small)
    - LM head: hidden_size * vocab_size (often tied with embeddings)
    """
    h = moe_config["hidden_size"]
    n_layers = moe_config["num_hidden_layers"]
    n_kv_heads = moe_config["num_kv_heads"]
    head_dim = moe_config["head_dim"]
    vocab = moe_config["vocab_size"]
    n_experts = moe_config["num_experts"]
    n_heads = h // head_dim if head_dim > 0 else 0

    if h == 0 or n_layers == 0:
        return {"error": "Missing dimensions for non-expert size calculation"}

    # Attention per layer: Q (h * h) + K (h * kv_dim) + V (h * kv_dim) + O (h * h)
    kv_dim = n_kv_heads * head_dim
    attn_params_per_layer = h * n_heads * head_dim + h * kv_dim + h * kv_dim + n_heads * head_dim * h
    # Norms per layer: 2 * h (pre-attn + post-attn RMSNorm)
    norm_params_per_layer = 2 * h
    # Router per layer: h * n_experts
    router_params_per_layer = h * n_experts

    total_per_layer = attn_params_per_layer + norm_params_per_layer + router_params_per_layer
    total_layers = total_per_layer * n_layers

    # Embeddings + LM head
    embed_params = vocab * h  # token embeddings
    lm_head_params = vocab * h  # may be tied

    total_params = total_layers + embed_params + lm_head_params

    sizes = {}
    for bits, label in [(16, "fp16"), (4, "4bit"), (3, "3bit"), (2, "2bit")]:
        total_bytes = total_params * bits / 8
        sizes[label] = {
            "total_params": total_params,
            "total_gb": round(total_bytes / (1024**3), 2),
        }

    return sizes


def audit_shard_layout(repo: str, index: dict | None) -> dict:
    """Analyze how expert tensors are distributed across safetensor shards."""
    if index is None:
        return {
            "total_shards": 0,
            "expert_tensor_count": 0,
            "non_expert_tensor_count": 0,
            "expert_shards": 0,
            "non_expert_shards": 0,
            "expert_groups": 0,
            "multi_shard_experts": 0,
            "contiguous_experts": True,
            "sample_expert_tensors": [],
            "error": "no_index",
        }

    weight_map = index.get("weight_map", {})
    if not weight_map:
        return {
            "total_shards": 0,
            "expert_tensor_count": 0,
            "non_expert_tensor_count": 0,
            "expert_shards": 0,
            "non_expert_shards": 0,
            "expert_groups": 0,
            "multi_shard_experts": 0,
            "contiguous_experts": True,
            "sample_expert_tensors": [],
            "error": "no_weight_map",
        }

    # Count unique shards
    shards = set(weight_map.values())

    # Find expert tensors and their shard distribution
    expert_tensors = {}
    non_expert_tensors = {}
    for tensor_name, shard_file in weight_map.items():
        # Expert tensors typically contain "experts" or "mlp.experts" in the name
        if "expert" in tensor_name.lower():
            expert_tensors[tensor_name] = shard_file
        else:
            non_expert_tensors[tensor_name] = shard_file

    # Check if experts from the same layer are in the same shard
    expert_shards = set(expert_tensors.values())
    non_expert_shards = set(non_expert_tensors.values())

    # Analyze per-expert contiguity
    # Group expert tensors by (layer, expert_id)
    expert_groups = {}
    for name, shard in expert_tensors.items():
        parts = name.split(".")
        layer_id = None
        expert_id = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                layer_id = parts[i + 1]
            if p == "experts" and i + 1 < len(parts):
                expert_id = parts[i + 1]
        if layer_id is not None and expert_id is not None:
            key = (layer_id, expert_id)
            if key not in expert_groups:
                expert_groups[key] = set()
            expert_groups[key].add(shard)

    # Check how many experts span multiple shards
    multi_shard_experts = sum(1 for shards_set in expert_groups.values() if len(shards_set) > 1)

    return {
        "total_shards": len(shards),
        "expert_tensor_count": len(expert_tensors),
        "non_expert_tensor_count": len(non_expert_tensors),
        "expert_shards": len(expert_shards),
        "non_expert_shards": len(non_expert_shards),
        "expert_groups": len(expert_groups),
        "multi_shard_experts": multi_shard_experts,
        "contiguous_experts": multi_shard_experts == 0,
        "sample_expert_tensors": list(expert_tensors.keys())[:10],
    }


def audit_model(repo: str, description: str) -> dict:
    """Run full audit on a single model."""
    print(f"\n{'='*60}")
    print(f"Auditing: {repo}")
    print(f"  {description}")
    print(f"{'='*60}")

    result = {"repo": repo, "description": description}

    # Fetch config
    config = fetch_config(repo)
    if config is None:
        result["error"] = "Cannot fetch config.json"
        return result

    # Extract MoE config
    moe_config = extract_moe_config(config)
    result["moe_config"] = moe_config
    print(f"\n  MoE config:")
    print(f"    Experts: {moe_config['num_experts']} total, {moe_config['num_experts_per_tok']} active per token")
    print(f"    Hidden: {moe_config['hidden_size']}, FFN: {moe_config['moe_intermediate_size']}")
    print(f"    Layers: {moe_config['num_hidden_layers']}, Vocab: {moe_config['vocab_size']}")

    # Estimate expert sizes
    expert_sizes = estimate_expert_sizes(moe_config)
    result["expert_sizes"] = expert_sizes
    if "error" not in expert_sizes:
        print(f"\n  Expert sizes (per expert):")
        for label, info in expert_sizes.items():
            print(f"    {label}: {info['mb_per_expert']:.1f} MB/expert, {info['total_expert_gb']:.1f} GB total ({info['total_experts']} experts)")

    # Estimate non-expert sizes
    non_expert_sizes = estimate_non_expert_size(moe_config)
    result["non_expert_sizes"] = non_expert_sizes
    if "error" not in non_expert_sizes:
        print(f"\n  Non-expert sizes:")
        for label, info in non_expert_sizes.items():
            print(f"    {label}: {info['total_gb']:.1f} GB ({info['total_params']:,} params)")

    # Memory budget check (9 GB model-side cap)
    if "error" not in non_expert_sizes:
        print(f"\n  Memory budget (9 GB model-side cap):")
        for label in ["4bit", "3bit", "2bit"]:
            ne_gb = non_expert_sizes[label]["total_gb"]
            kv_gb = 1.0  # conservative KV cache estimate
            scratch_gb = 0.5
            runtime_gb = 0.5
            total = ne_gb + kv_gb + scratch_gb + runtime_gb
            fits = total < 9.0
            status = "FITS" if fits else "EXCEEDS"
            print(f"    {label}: {ne_gb:.1f} + {kv_gb} + {scratch_gb} + {runtime_gb} = {total:.1f} GB [{status}]")
            result[f"fits_{label}"] = fits

    # Shard layout
    index = fetch_model_index(repo)
    shard_layout = audit_shard_layout(repo, index)
    result["shard_layout"] = shard_layout
    print(f"\n  Shard layout:")
    print(f"    Total shards: {shard_layout['total_shards']}")
    print(f"    Expert tensors: {shard_layout['expert_tensor_count']}")
    print(f"    Non-expert tensors: {shard_layout['non_expert_tensor_count']}")
    print(f"    Experts contiguous: {shard_layout['contiguous_experts']}")
    if shard_layout['multi_shard_experts'] > 0:
        print(f"    WARNING: {shard_layout['multi_shard_experts']} experts span multiple shards")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    env = get_environment_info()
    print(f"Hardware: {env['chip']}, {env['memory_gb']} GB, available: {env['available_gb']:.1f} GB")
    print(f"macOS {env['macos_version']}")

    all_results = []

    # Audit full-precision models (for architecture info)
    print("\n" + "=" * 60)
    print("FULL-PRECISION MODEL AUDITS")
    print("=" * 60)
    for model in MOE_MODELS:
        result = audit_model(model["repo"], model["description"])
        result["preferred"] = model["preferred"]
        all_results.append(result)

    # Audit MLX quantized variants
    print("\n" + "=" * 60)
    print("MLX QUANTIZED MODEL AUDITS")
    print("=" * 60)
    for model in MLX_MOE_MODELS:
        result = audit_model(model["repo"], model["description"])
        result["preferred"] = model["preferred"]
        all_results.append(result)

    # Summary and recommendation
    print("\n" + "=" * 60)
    print("SUMMARY & RECOMMENDATION")
    print("=" * 60)

    viable = []
    for r in all_results:
        if r.get("error"):
            print(f"  SKIP {r['repo']}: {r['error']}")
            continue
        for label in ["4bit", "3bit", "2bit"]:
            if r.get(f"fits_{label}"):
                viable.append((r["repo"], label, r.get("preferred", False)))

    if viable:
        print(f"\n  Viable models (fit in 9 GB model-side cap):")
        for repo, quant, preferred in viable:
            pref = " [PREFERRED]" if preferred else ""
            print(f"    {repo} @ {quant}{pref}")
        # Pick recommendation
        preferred_viable = [v for v in viable if v[2]]
        if preferred_viable:
            rec = preferred_viable[0]
        else:
            rec = viable[0]
        print(f"\n  RECOMMENDATION: {rec[0]} @ {rec[1]}")
    else:
        print("\n  WARNING: No models fit within 9 GB model-side cap!")
        print("  Consider more aggressive quantization or smaller models.")

    # Log all results
    log_experiment(
        experiment_name="checkpoint_audit",
        phase="checkpoint_audit",
        config={"models_audited": [m["repo"] for m in MOE_MODELS + MLX_MOE_MODELS]},
        results={
            "audits": all_results,
            "viable_models": viable,
            "recommendation": rec if viable else None,
        },
        env=env,
    )

    return all_results


if __name__ == "__main__":
    main()
