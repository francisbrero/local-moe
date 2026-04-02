# H8: Safetensors Direct Streaming — Zero-Conversion Block Loading

**Status**: Untested
**Analogy**: Memory-mapped file I/O (mmap)
**Bottleneck addressed**: Serialization overhead (the 750x gap between raw I/O and actual block swap)

## The Insight

Database engines don't deserialize their storage format to serve queries — they memory-map the on-disk format directly and let the CPU read it in place. The OS page cache handles caching transparently, and the data is never copied or converted.

The SSD Layer LOD experiment (H0+H5) proved that streaming 72B transformer blocks from NVMe works mechanically — memory stays stable, no OOM, no thrashing. But it also revealed that the **serialization layer** (numpy npz) is 750x slower than the raw I/O. Each 471 MB block requires: np.load → np array → mx.array copy → mx.eval sync. The numpy round-trip adds 13x overhead on top of the raw SSD read.

Safetensors files are designed for zero-copy memory mapping. The weight tensors are stored in a flat binary layout that can be mmap'd directly — no parsing, no conversion, no intermediate copies.

## Hypothesis

> By loading streaming blocks directly from safetensors shards via memory-mapped I/O instead of re-serialized npz files, we can eliminate the 13x serialization overhead and achieve 0.5-2 tok/s on Qwen2.5-72B with 24 GB RAM — bringing SSD layer streaming into the usable range.

## Mechanism

```
Current path (H0+H5, Phase 3):
  model.safetensors → mx.load() → mx.eval() → np.savez() → disk
  disk → np.load() → mx.array() → mx.eval() → forward
  Latency: 704 ms/block (471 MB)

Proposed path:
  model.safetensors → mmap'd directly (no intermediate save)
  Per-token: offset into shard → mmap read → mx.array(view) → forward
  Expected: ~55-80 ms/block (matching Phase 2b raw I/O)
```

### Implementation sketch

1. **Build a block-to-shard index**: At startup, scan safetensors shard headers to map each transformer block's weight tensors to (shard_file, byte_offset, byte_length) tuples. This is metadata-only — no weights loaded.

2. **Direct mmap loading**: For each streaming block, open the safetensors shard, seek to the tensor offsets, and create MLX arrays from the raw memory. Safetensors guarantees little-endian, aligned, contiguous storage.

3. **Leverage MLX's native safetensors support**: `mx.load("model.safetensors")` already uses mmap internally. The key is to load only the tensors for a single block, not the whole file. MLX's `mx.load` accepts a list of keys — we can load per-block subsets.

4. **Skip the save step entirely**: Instead of saving blocks to npz during setup, keep a reference to the original safetensors shards. The OS page cache will naturally keep hot blocks resident.

## Expected Impact

| Metric | Phase 3 (npz) | H8 (safetensors) | Improvement |
|--------|---------------|-------------------|-------------|
| Block swap latency | 704 ms | 55-80 ms | 9-13x |
| Per-token time | 188,000 ms | 3,500-5,000 ms | 38-54x |
| tok/s | 0.005 | 0.2-0.3 (cold) | 40-60x |
| tok/s (warm cache) | — | 0.5-2.0 | new |

These estimates assume:
- 64 streaming blocks × 55-80 ms I/O each = 3.5-5.1s per token (cold)
- With 9 GB page cache at 55% occupancy, ~36 blocks cached → ~28 from SSD → ~1.5-2.2s → 0.5-0.7 tok/s
- Higher cache hit rates (sequential access, Zipf-like reuse) could push to 1-2 tok/s

## Why This Might Work

1. **Phase 2b proved raw I/O is fast**: 262 MB blocks read at 55 ms via pread — the SSD is not the bottleneck
2. **Safetensors is designed for mmap**: Flat binary layout, no parsing overhead, aligned tensors
3. **MLX already supports safetensors mmap**: `mx.load()` uses mmap internally — we just need per-block granularity
4. **OS page cache is the right abstraction**: H0 showed 63-78% cache residency with Zipf access patterns. Sequential block access (layer 0→1→...→79 every token) is even more cache-friendly
5. **No intermediate format needed**: Loading directly from the model's own safetensors shards eliminates the save-to-disk step entirely

## What This Enables

| Scenario | RAM needed | Speed (est.) | Notes |
|----------|-----------|-------------|-------|
| 72B on 24 GB (8+8 Q4/64 Q2) | 14.9 GB pinned | 0.5-2 tok/s | Primary target |
| 72B on 24 GB (3+5 Q4/72 Q2) | 11.2 GB pinned | 0.3-1.5 tok/s | More cache headroom |
| 72B on 16 GB (all streamed Q2) | 6.9 GB pinned | 0.1-0.5 tok/s | Extreme streaming |

For comparison: 0.5 tok/s on 72B is slow but usable for batch/offline tasks. 1-2 tok/s approaches interactive use for short responses.

## Experiment Plan

### Phase 1: Safetensors block index
- Parse safetensors shard headers for Qwen2.5-72B
- Build block → (shard, offset, length) mapping
- Verify total sizes match Phase 0 budget numbers

### Phase 2: Direct load on 7B
- Replace npz save/load with safetensors direct loading
- Benchmark on Qwen2.5-7B: compare npz path vs safetensors path
- Measure per-block latency, RSS stability, logit correctness

### Phase 3: 72B integration
- Apply to 72B with 8+8 Q4 / 64 Q2 config
- Measure tok/s, memory stability, output quality
- Compare to Phase 3 npz results (0.005 tok/s baseline)

### Phase 4: Cache optimization
- Test MADV_SEQUENTIAL on safetensors shards
- Measure page cache hit rate over sustained generation
- Explore readahead tuning for sequential block access

## Open Questions

1. Can we load individual tensor subsets from a safetensors shard without loading the entire file? MLX's `mx.load` may load everything — need to check granularity.
2. Does mmap of large safetensors shards (8-10 GB each) cause page table pressure on macOS?
3. How does quantization metadata (scales, biases) align in the safetensors layout? QuantizedLinear needs weight + scales + biases as separate tensors.
4. Can we combine this with pre-quantized Q2 safetensors for the streaming blocks to halve the I/O per block (262 MB vs 471 MB)?
5. What's the interaction with MLX's lazy evaluation? Can we avoid the mx.eval barrier by letting MLX batch the forward pass across multiple loaded blocks?

## Risks

- MLX's `mx.load` may copy mmap'd data internally (similar to Phase 1b finding) — need to verify that safetensors path avoids the numpy round-trip
- Large mmap regions may compete with the page cache needed for block streaming
- Safetensors shard boundaries may not align with transformer block boundaries — one block's weights may span multiple shards
- Output quality issues from Phase 3 (degenerate text) may not be serialization-related — could indicate a bug in weight restoration logic
