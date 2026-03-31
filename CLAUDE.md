# local-moe

## Project Goal

Run the largest possible LLM inference on a **16GB M4 MacBook Pro** using techniques from Flash MOE, TurboQuant, and related research. We go as low-level as needed — custom Metal shaders, C, whatever works.

## Hardware

- Apple M4, 16GB unified memory (~10-11GB usable)
- NVMe SSD ~5-7 GB/s read
- Metal 4 GPU, Apple Accelerate framework
- No discrete GPU — unified memory, zero-copy CPU/GPU

## Key Files

- `Agents.md` — 10 research approaches with experiment plan
- `dev/active/` — Current experiment notes (gitignored)
- `dev/completed/` — Archived experiments (gitignored)
- `.claude/skills/` — Technical skills and runbooks

## Tech Stack

- **Languages**: C, Metal Shading Language (MSL), Python, Swift (as needed)
- **Frameworks**: MLX, llama.cpp, Metal, Accelerate
- **Quantization**: GGUF, BitNet, QuIP#, AQLM, TurboQuant
- **Build**: Make / CMake for C/Metal, uv for Python

## Commands

```bash
# Python environment
uv sync
uv run python <script>

# GitHub
gh issue list
gh issue view <num>
gh pr create

# Build (once we have C/Metal code)
make build
make test
make bench
```

## Workflow

1. Pick an issue from GitHub (`gh issue list --label high-priority`)
2. Create a branch (`feature/<issue-slug>` or `experiment/<issue-slug>`)
3. Document experiment plan in `dev/active/<name>/plan.md`
4. Implement, benchmark, log results
5. Update `dev/active/<name>/context.md` with findings
6. PR with results, move to `dev/completed/` when done

## Conventions

- Experiments are cheap and disposable — try things fast, measure, iterate
- Always benchmark before and after changes
- Log metrics: tok/s, peak memory (RSS), GPU memory, cache hit rate, perplexity
- Commit working benchmarks even if results are poor — negative results are valuable
- Keep Metal shaders in `src/metal/`, C code in `src/`, Python scripts in `scripts/`
- Use `experiments.jsonl` for structured experiment logging

## Code Style

- C: follow llama.cpp conventions (snake_case, minimal dependencies)
- Metal: one kernel per .metal file, descriptive names
- Python: ruff for formatting, type hints encouraged
- Commit messages: imperative mood, reference issue number

## Skill Auto-Activation

Skills in `.claude/skills/` are automatically surfaced based on prompt keywords. See `.claude/skills/skill-rules.json` for trigger configuration.
