---
name: cross-revision-benchmark
description: >-
  Benchmark the same Cytnx function at two or more git revisions and produce a
  fair comparison table using the build-test-workflow script. Use when a
  change is motivated by performance, a reviewer questions a speedup/
  regression, or a PR needs before/after numbers — e.g. comparing a branch tip
  against its merge base. Covers picking comparison points that already have
  the benchmark committed, reusing one build dir switched sequentially
  between revisions in place (never a fresh worktree or a separate build dir
  per revision), and reporting rules (same machine/session, commit hash plus
  a rebase-proof readable label, stated aggregation).
---

# Cross-revision benchmarking

Performance claims are settled by measurement (never plausibility), and a
measurement is only meaningful when every column ran under identical
conditions. This skill is the procedure for benchmarking the *same* code path
at several git revisions on one machine, in one session, building and running
through `build-test-workflow`'s script:

```sh
S=.claude/skills/build-test-workflow/scripts/build_preset.sh
```

Only benchmark a **committed** Google Benchmark file (`benchmarks/*.cpp`,
built as the `benchmarks_main` target), and every revision being compared
must already have it registered in `benchmarks/CMakeLists.txt`. This follows
automatically from committing the benchmark before the change it measures:
pick the earliest comparison point as the commit where the benchmark itself
landed, not further back — or, if the benchmark is already on the base
branch, the base branch tip directly. If no existing file covers the
function under comparison, add it (and commit it) first, before the change
being measured — do not write comparison code in a scratch harness; a fixed,
versioned benchmark source is what makes a fair cross-revision comparison
possible at all.

Use a **release** preset (`openblas-cpu` or similar) for every column — a
debug/ASan build's timings are not representative. `"$S"` strips `--coverage`
instrumentation from every build it produces (see `build-test-workflow`), so
a dir it already built for `test_main`/`pycytnx` is safe to reuse for
`benchmarks_main` too.

## Procedure

Switch revisions **in place, in the main working tree** — do not create a
worktree. A fresh worktree starts with an empty build dir, so its first build
would be a full rebuild no matter how small the diff between revisions is;
checking out in place instead reuses whatever `build/<preset>` the script has
already produced from ordinary development. A whole-tree `git checkout <rev>`
only rewrites files whose blob content actually changed, so the reused dir
stays incremental across every switch — verified: switching between two
commits with a several-hundred-line diff, including one newly-added file,
left only a handful of build steps pending in an already-built dir. Stay
strictly sequential: never build or run more than one revision at a time, and
never run a benchmark alongside another build — CPU contention skews results.

Before starting: `git status`, and stash (`git stash -u`) any uncommitted
work; note the current branch/ref to return to at the end.

For each revision, in order:

1. `git checkout <rev>` (whole tree) — the benchmark file is already present
   and registered, so nothing else to prepare.
2. Build and run:

   ```sh
   "$S" openblas-cpu --target benchmarks_main --test \
     --benchmark_repetitions=5 --benchmark_report_aggregates_only=true \
     --benchmark_filter='<pattern>' > /tmp/bm-<label>.txt
   ```

When every column is recorded, return to where you started: `git checkout
<original-branch>` and `git stash pop` if step 0 created a stash.

## Reporting rules

- Every column needs **both** the exact commit hash **and** a short, stable,
  readable label ("master", "merge-base", "min-heap fix") — a hash alone goes
  stale the moment history is rewritten (rebase, squash, force-push), while
  the label is what the comparison is actually *about* and survives that.
  State both together, e.g. "min-heap fix (`5d4ca56`)".
- All columns fresh in the **same session, same machine, same build flags**.
- State the aggregation ("mean of 5 repetitions") and the unit.
- Compare against a **correct** baseline. If an old revision is buggy in a way
  that changes how much work it does (e.g. out-of-bounds reads causing
  skipped work), its numbers are not a correctness-preserving baseline — say
  so explicitly rather than letting the table imply a fair race.
- Record the deciding numbers in the PR description or commit message so the
  performance choice is auditable.
