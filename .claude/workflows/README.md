# Working on this repository with a multi-agent (ultracode) session

This folder holds saved Claude Code workflows for this repository. A workflow is a script that
runs many agents with a fixed structure (review, then verify, then summarize). Everything a
session needs to know about the repository is in the root `CLAUDE.md`, which every agent loads.

## Before you start a session

1. **Commit the working tree first** (on a branch). Agents that are given their own git
   worktree start from committed history, by default from the remote default branch. Work
   that is not committed is invisible to them. If the branch is not pushed, set
   `worktree.baseRef` to `head` in your Claude Code settings so worktrees start from your
   local HEAD (check the current name of that setting in the Claude Code worktrees
   documentation before relying on it).
2. **Confirm the baseline is green**: `python run_tier1.py` must end with `0 failed`.
3. **Use the Python 3.11 interpreter** named in `CLAUDE.md`. The `python` on PATH on the
   original workstation is a different version.
4. Optional: allow the read-only and verification commands you expect agents to run
   (`python run_tier1.py`, `python -m salmonid.…`, `git status`, `git diff`, `git show`) in your
   permission settings, so a long run does not pause on permission prompts. A run pauses on
   its own only for agent permission prompts.

## Turning ultracode on

- For one task: include the word `ultracode` in your prompt.
- For the whole session: `/effort ultracode` (off again with `/effort ultracode off`).
- The "Dynamic workflow size" row in `/config` sets how many agents a workflow should aim for
  (`small` under 5, `medium` under 10, `large` under 50, `unrestricted`). The audit below uses
  about 10 reviewers plus one verifier per finding, so `large` suits it.

## Saved workflows

| Command | What it does | Changes files? |
|---|---|---|
| `/deposit-audit` | Independent reproducibility audit. Runs `run_tier1.py`, sends one reviewer to each area of the repository, has every finding checked by an agent instructed to refute it, drops anything already recorded in `docs/REPRODUCIBILITY_REVIEW.md`, and returns the confirmed new findings as draft register entries. | No (read-only) |

Run a subset by passing the areas as arguments, for example
`/deposit-audit ["salmonid/count", "salmonid/results"]`.

After editing a workflow file, run `/reload-skills`.

## Suggested first prompts for an ultracode session

The register of open work is `docs/REPRODUCIBILITY_REVIEW.md`. Items marked **AUTHOR** need a
human decision and should not be given to agents. Good candidates for a session:

1. "ultracode: run /deposit-audit and add the confirmed findings to the register."
2. "ultracode: for every item in docs/PAPER_TO_CODE.md with status NO CODE or MISMATCH, work out
   what is needed to close it, and close the ones that need no author decision. Keep
   `python run_tier1.py` green."
3. "ultracode: write a unit test for each statistic in `salmonid/results/` against a
   hand-computed case (Lin's CCC, Wilson interval, moving-block resample, Bland-Altman
   limits), without changing the modules."
4. "ultracode: check every claim in the manuscript's Methods against the code and list the
   sentences the code does not support" (read-only on the manuscript repository).
