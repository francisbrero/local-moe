Fix GitHub issue #$ARGUMENTS

First, validate that "$ARGUMENTS" is a numeric issue number. If not, stop and ask the user for a valid issue number.

The guiding principle of this workflow is **context persistence + automated quality gates**: every phase checkpoints to `dev/active/<slug>/` so work survives across sessions, and plan/code reviews loop until they converge on no material findings. Delegate open-ended exploration to subagents; keep targeted lookups (known file/symbol) in the main context.

## Phase 0: Resume Detection

1. Look for an existing dev-docs folder for this issue:
   - First match `dev/active/issue-$ARGUMENTS-*`.
   - If none, search `dev/active/*/context.md` for one referencing issue #$ARGUMENTS.
2. If a folder is found:
   - Read its `context.md` to restore the checkpoint (current phase, key files, review rounds, next steps).
   - Read `tasks.md` and skip already-completed steps.
   - Confirm with the user: "Found in-progress work for issue #$ARGUMENTS at `<path>` (last step: <step>). Resume from here?" — resume on yes, start fresh on no.
3. If no folder is found, proceed to Phase 1 as a fresh start.

## Phase 1: Setup

1. Retrieve issue details: `gh issue view $ARGUMENTS`
2. Read the issue description, labels, and any linked references. Clarify ambiguous requirements with the user before planning.
3. Determine the branch prefix from the issue type:
   - `experiment/<slug>` for research issues
   - `feature/<slug>` for infra/tooling issues
   - `bugfix/<slug>` for bug fixes
4. Create the branch from master. If the branch already exists, ask the user whether to reuse it or pick a new name.
5. Create dev-docs at `dev/active/issue-$ARGUMENTS-<slug>/` (issue-number prefix enables Phase 0 resume):
   - `plan.md` — approach and what we're testing
   - `context.md` — current state, findings, next steps, review-round counts
   - `tasks.md` — checklist of sub-tasks with completion status
6. If working in a git worktree, install the Python environment (`uv sync`) — worktrees don't inherit the gitignored `.venv`.

## Phase 2: Plan + Automated Review Loop

1. Draft the plan in `dev/active/issue-$ARGUMENTS-<slug>/plan.md`, broken into phases. Record the sub-tasks (with dependency order) in `tasks.md` and, for multi-step work, mirror them with TaskCreate for dependency tracking.
2. Run an automated review loop (max 10 rounds). Each round:
   a. Use the Agent tool to spawn a subagent with `subagent_type: "general-purpose"` and `model: "sonnet"`. In the prompt, tell the agent: "Read the instructions in `.claude/agents/plan-reviewer.md` and follow them. PLAN_PATH=`dev/active/issue-$ARGUMENTS-<slug>/plan.md`, ISSUE_NUMBER=`$ARGUMENTS`."
   b. Read the subagent's response and look for a JSON block with `material_findings`, `findings`, and `summary` fields.
   c. If `material_findings` is `true`: address each high/medium severity finding by updating the plan, then continue to the next round.
   d. If `material_findings` is `false` (i.e. `MATERIAL_FINDINGS: false`), or the response contains `"error": true`, or JSON parsing fails: break out of the loop.
3. Present the reviewed plan to the user. Use AskUserQuestion with these options:
   - "Approve plan" — proceed to implementation
   - "Request changes" — user provides feedback to incorporate
   - "Skip to implementation" — bypass further review
4. If the user requests changes, incorporate their feedback and optionally re-run the review loop.
5. Do not proceed to Phase 3 until the plan is approved or the user chooses to skip.
6. Update `context.md` with the plan-review round count.

## Phase 3: Implementation

1. Implement the solution following the approved plan, ticking off `tasks.md` (and TaskUpdate) as steps complete.
2. Run benchmarks as needed.
3. Log results to `experiments.jsonl` if this is a research experiment.
4. Ensure any tests pass before committing.
5. Checkpoint `context.md` after meaningful progress so the session is resumable.

## Phase 4: Automated Code Review Loop

1. Commit work-in-progress changes (so Codex can review committed diffs).
2. Run an automated review loop (max 10 rounds). Each round:
   a. Use the Agent tool to spawn a subagent with `subagent_type: "general-purpose"` and `model: "sonnet"`. In the prompt, tell the agent: "Read the instructions in `.claude/agents/code-reviewer.md` and follow them."
   b. Read the subagent's response and look for a JSON block with `material_findings`, `findings`, and `summary` fields.
   c. If `material_findings` is `true`: fix each high/medium severity finding, commit the fixes, then continue to the next round.
   d. If `material_findings` is `false` (i.e. `MATERIAL_FINDINGS: false`), or the response contains `"error": true`, or JSON parsing fails: break out of the loop.

## Phase 5: Finalize

1. Ensure all changes are committed with a message referencing the issue: `Fixes #$ARGUMENTS`.
2. Update `dev/active/issue-$ARGUMENTS-<slug>/context.md` with:
   - Final findings and results
   - Review stats: number of plan review rounds, code review rounds, total findings addressed
3. Create a PR with `gh pr create`, filling the description from the issue and git context and including the review round counts. Respect any existing PR template in the repo.
4. On merge, move the dev-docs folder from `dev/active/` to `dev/completed/`.
