Fix GitHub issue #$ARGUMENTS

First, validate that "$ARGUMENTS" is a numeric issue number. If not, stop and ask the user for a valid issue number.

## Phase 0: Setup

1. Retrieve issue details: `gh issue view $ARGUMENTS`
2. Read the issue description, labels, and any linked references
3. Determine the branch prefix from the issue type:
   - `experiment/<slug>` for research issues
   - `feature/<slug>` for infra/tooling issues
   - `bugfix/<slug>` for bug fixes
4. Create the branch from master. If the branch already exists, ask the user whether to reuse it or pick a new name.
5. Create experiment docs in `dev/active/<slug>/`:
   - `plan.md` — approach and what we're testing
   - `context.md` — current state, findings, next steps
   - `tasks.md` — checklist of sub-tasks

## Phase 1: Plan + Automated Review Loop

1. Draft the plan in `dev/active/<slug>/plan.md`
2. Run an automated review loop (max 10 rounds). Each round:
   a. Use the Agent tool to spawn a subagent with `subagent_type: "general-purpose"` and `model: "sonnet"`. In the prompt, tell the agent: "Read the instructions in `.claude/agents/plan-reviewer.md` and follow them. PLAN_PATH=`dev/active/<slug>/plan.md`, ISSUE_NUMBER=`$ARGUMENTS`."
   b. Read the subagent's response and look for a JSON block with `material_findings`, `findings`, and `summary` fields.
   c. If `material_findings` is `true`: address each high/medium severity finding by updating the plan, then continue to the next round.
   d. If `material_findings` is `false`, or the response contains `"error": true`, or JSON parsing fails: break out of the loop.
3. Present the reviewed plan to the user. Use AskUserQuestion with these options:
   - "Approve plan" — proceed to implementation
   - "Request changes" — user provides feedback to incorporate
   - "Skip to implementation" — bypass further review
4. If the user requests changes, incorporate their feedback and optionally re-run the review loop.
5. Do not proceed to Phase 2 until the plan is approved or the user chooses to skip.

## Phase 2: Implementation

1. Implement the solution following the approved plan
2. Run benchmarks as needed
3. Log results to `experiments.jsonl` if this is a research experiment
4. Ensure any tests pass before committing

## Phase 3: Automated Code Review Loop

1. Commit work-in-progress changes (so Codex can review committed diffs)
2. Run an automated review loop (max 10 rounds). Each round:
   a. Use the Agent tool to spawn a subagent with `subagent_type: "general-purpose"` and `model: "sonnet"`. In the prompt, tell the agent: "Read the instructions in `.claude/agents/code-reviewer.md` and follow them."
   b. Read the subagent's response and look for a JSON block with `material_findings`, `findings`, and `summary` fields.
   c. If `material_findings` is `true`: fix each high/medium severity finding, commit the fixes, then continue to the next round.
   d. If `material_findings` is `false`, or the response contains `"error": true`, or JSON parsing fails: break out of the loop.

## Phase 4: Finalize

1. Ensure all changes are committed with a message referencing the issue: `Fixes #$ARGUMENTS`
2. Update `dev/active/<slug>/context.md` with:
   - Final findings and results
   - Review stats: number of plan review rounds, code review rounds, total findings addressed
3. Create a PR with `gh pr create`, including review round counts in the PR description
