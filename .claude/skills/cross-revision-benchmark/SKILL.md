---
name: cross-revision-benchmark
description: >-
  Use whenever a performance claim must be made or checked: a change is
  motivated by speed, a reviewer questions a speedup or regression, a PR
  needs before/after numbers, or two implementations/revisions must be
  timed against each other. Never assert performance from reasoning or a
  scratch timing loop — run this skill's benchmark procedure instead.
---

# Cross-revision benchmarking

Compare two columns by default: the branch **tip** and its **base**
(`git merge-base HEAD master`).

## The benchmark comes first

Benchmark only a committed Google Benchmark file (`benchmarks/*.cpp`,
registered in `benchmarks/CMakeLists.txt`, built as `benchmarks_main`).
Commit the benchmark as the branch's first commit, before the change it
measures — then it exists at every revision being compared. Pick the base
column accordingly: the commit where the benchmark landed, or the base
branch tip directly if the benchmark is already there.

## Running a column (no rebuild churn)

Use a **release** preset (`openblas-cpu` or similar) — debug/ASan timings
are not representative. `build_preset.sh` builds are uninstrumented:
`RUN_TESTS=ON` would add `--coverage` to the cytnx library and skew the
timings of anything linking it, so the script wires in
`strip-coverage-launcher.sh` as the compiler/linker launcher automatically
(see that script for details) — a build dir it made for tests is safe to
reuse for benchmarks.

Switch revisions in place in the main tree (`git checkout <rev>`) — never a
fresh worktree, which starts with an empty build dir and forces a full
rebuild; in-place checkout keeps `build/<preset>` incremental. Stash
uncommitted work first and return to the original ref at the end. Per
revision:

```sh
S=.claude/skills/build-test-workflow/scripts/build_preset.sh
"$S" openblas-cpu --target benchmarks_main --test \
  --benchmark_repetitions=5 --benchmark_report_aggregates_only=true \
  --benchmark_filter='<pattern>' > /tmp/bm-<label>.txt
```

**Strictly sequential:** never run a benchmark concurrently with another
build or benchmark run — CPU contention skews results.

## Report

- Every column: a readable label **and** its commit SHA — e.g. "min-heap
  fix (`5d4ca56`)". All columns from the same machine and session; state
  the aggregation ("mean of 5 repetitions") and the unit.
- Compare against a **correct** baseline: if an old revision is buggy in a
  way that changes how much work it does (e.g. out-of-bounds reads causing
  skipped work), its numbers are not a fair baseline — say so explicitly
  rather than letting the table imply a fair race.
- Append the deciding numbers to the PR description so the performance
  choice is auditable.
