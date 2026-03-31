---
description: MLX framework patterns for Apple Silicon inference
globs:
  - "scripts/**/*.py"
  - "**/*mlx*"
alwaysApply: false
---

# MLX Inference Patterns

## Setup

```bash
uv add mlx mlx-lm
```

## Key Techniques

### Mixed-Precision Quantization
```bash
# 4-bit body, 6-bit embeddings for better quality
mlx_lm.convert --hf-path <model> --mlx-path <output> \
    -q --q-bits 4 --q-group-size 64
```

### Kernel Fusion
```python
import mlx.core as mx

@mx.compile
def fused_op(x, w, b):
    return mx.fast.rms_norm(x @ w + b, weight=None, eps=1e-5)
```

### Fast Operations
Use `mx.fast` for optimized implementations:
- `mx.fast.rms_norm` — fused RMS normalization
- `mx.fast.scaled_dot_product_attention` — optimized attention

### Unified Memory
MLX operations run on CPU or GPU transparently. No explicit device transfers needed on Apple Silicon.

## Benchmarking

```python
import time
import mlx.core as mx

mx.eval(output)  # force sync
start = time.perf_counter()
# ... inference ...
mx.eval(output)
elapsed = time.perf_counter() - start
```

## Resources

- See issue #11 for MLX evaluation plan
- [MLX GitHub](https://github.com/ml-explore/mlx)
