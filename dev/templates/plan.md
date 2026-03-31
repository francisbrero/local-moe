# Experiment: [NAME]

**Issue**: #[NUMBER]

## Hypothesis

What do we expect to happen?

## Approach

How will we test this?

## Metrics

What will we measure?

- [ ] tok/s (decode)
- [ ] ttft (time to first token)
- [ ] peak_rss_mb
- [ ] gpu_memory_mb
- [ ] cache_hit_rate (if applicable)
- [ ] perplexity

## Baseline

What are we comparing against? Include the benchmark command:
```bash
# e.g. uv run python scripts/benchmark.py --config baseline.json
```

## Success Criteria

What result would make this approach worth pursuing further?

## Rollback

What do we revert to if this doesn't work?
