Fix GitHub issue #$ARGUMENTS

## Steps

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
5. Present the plan and wait for approval before implementing
6. Implement the solution, running benchmarks as needed
7. Log results to `experiments.jsonl` if this is a research experiment
8. Ensure any tests pass before committing
9. Commit with a message referencing the issue: `Fixes #<number>`
10. Create a PR: `gh pr create`
