Fix GitHub issue #$ARGUMENTS

## Phase 0: Setup

1. Retrieve issue details: `gh issue view $ARGUMENTS`
2. Read the issue description, labels, and any linked references
3. Create a branch from master:
   - `experiment/<slug>` for research issues
   - `feature/<slug>` for infra/tooling issues
   - `bugfix/<slug>` for bug fixes
4. Create experiment docs in `dev/active/<slug>/`:
   - `plan.md` — approach and what we're testing
   - `context.md` — current state, findings, next steps
   - `tasks.md` — checklist of sub-tasks

## Phase 1: Plan + Automated Review Loop

1. Draft the plan in `dev/active/<slug>/plan.md`
2. **Automated review loop** (max 10 rounds):
   a. Spawn the `plan-reviewer` subagent using the Agent tool:
      - `subagent_type: "general-purpose"`
      - Prompt: "You are the plan-reviewer agent. PLAN_PATH=dev/active/<slug>/plan.md, ISSUE_NUMBER=$ARGUMENTS. Follow the instructions in .claude/agents/plan-reviewer.md"
      - `model: "sonnet"`
   b. Parse the JSON result from the subagent
   c. If `material_findings` is `true`:
      - Address each high/medium severity finding by updating the plan
      - Log the round number and findings count
      - Continue to next round
   d. If `material_findings` is `false` or `error` is `true`:
      - Break out of the loop
3. Present the reviewed plan to the user via `AskUserQuestion`:
   - Include a summary of review rounds completed and findings addressed
   - Options: "Approve plan", "Request changes", "Skip to implementation"
4. If user requests changes, incorporate feedback and optionally re-run review loop
5. Do not proceed to Phase 2 until the plan is approved

## Phase 2: Implementation

1. Implement the solution following the approved plan
2. Run benchmarks as needed
3. Log results to `experiments.jsonl` if this is a research experiment
4. Ensure any tests pass before committing

## Phase 3: Automated Code Review Loop

1. Commit work-in-progress changes (so Codex can review committed diffs)
2. **Automated review loop** (max 10 rounds):
   a. Spawn the `code-reviewer` subagent using the Agent tool:
      - `subagent_type: "general-purpose"`
      - Prompt: "You are the code-reviewer agent. Follow the instructions in .claude/agents/code-reviewer.md"
      - `model: "sonnet"`
   b. Parse the JSON result from the subagent
   c. If `material_findings` is `true`:
      - Fix each high/medium severity finding
      - Commit the fixes
      - Log the round number and findings count
      - Continue to next round
   d. If `material_findings` is `false` or `error` is `true`:
      - Break out of the loop

## Phase 4: Finalize

1. Ensure all changes are committed with a message referencing the issue: `Fixes #$ARGUMENTS`
2. Update `dev/active/<slug>/context.md` with:
   - Final findings and results
   - Review stats: number of plan review rounds, code review rounds, total findings addressed
3. Create a PR: `gh pr create`
   - Include review round counts in the PR description
