---
name: cross-revision-benchmark
description: >-
  Benchmark the same Cytnx function at two or more git revisions and produce a
  fair comparison table using the build-test-workflow script. Use when a
  change is motivated by performance, a reviewer questions a speedup/
  regression, or a PR needs before/after numbers — e.g. comparing a branch tip
  against its merge base. Covers the fixed-benchmark-source vs
  per-revision-benchmark-source cases, reusing one build dir switched
  sequentially between revisions in place (never a fresh worktree or a
  separate build dir per revision), and reporting rules (same machine/
  session, commit hash plus a rebase-proof readable label, stated
  aggregation).
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
built as the `benchmarks_main` target). If no existing file covers the
function under comparison, add one first — do not write comparison code in a
scratch harness; a fixed, versioned benchmark source is what makes a fair
cross-revision comparison possible at all. Use a **release** preset
(`openblas-cpu` or similar) for every column — a debug/ASan build's timings
are not representative.

## Which of two cases applies

- **Case A — the benchmark's call signature is unchanged across every
  revision.** Keep **one fixed benchmark source** (the tip's) and vary only
  the library underneath.
- **Case B — the function's public interface changed** between revisions, so
  one benchmark source cannot compile against all of them (or a revision
  being compared predates the benchmark file existing at all). Use **each
  revision's own benchmark source** instead of forcing one file across all
  columns.

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

1. `git checkout <rev>` (whole tree).
2. **Case A only:** overlay the tip's benchmark file **and its build
   registration** on top — `benchmarks/CMakeLists.txt` lists every `.cpp` in
   the target's sources, so overlaying only the `.cpp` silently leaves it
   uncompiled if this revision predates the file existing at all:

   ```sh
   git checkout <tip> -- benchmarks/<the_bm_file>.cpp benchmarks/CMakeLists.txt
   ```

   **Case B:** skip this — build whatever benchmark file already exists at
   this revision.
3. Build and run:

   ```sh
   "$S" openblas-cpu --target benchmarks_main --test \
     --benchmark_repetitions=5 --benchmark_report_aggregates_only=true \
     --benchmark_filter='<pattern>' > /tmp/bm-<label>.txt
   ```

4. **Case A only:** restore the overlay before moving to the next revision —
   the working tree still holds `<tip>`'s content for the overlaid files
   after step 2, so `git checkout <rev>` in the next iteration's step 1
   would otherwise either refuse to switch or (if it can 3-way-merge) carry
   `<tip>`'s content silently forward onto the wrong revision:

   ```sh
   git restore --source=HEAD --staged --worktree -- \
     benchmarks/<the_bm_file>.cpp benchmarks/CMakeLists.txt
   ```

   Use `git restore`, not `git checkout HEAD -- <path>`: when `<rev>`
   predates the benchmark file existing at all, the overlay added a path
   HEAD doesn't have, and `checkout HEAD -- <path>` errors with "did not
   match any file(s) known to git" for a path absent from HEAD, whereas
   `restore --source=HEAD` correctly deletes it to match HEAD's (its
   absence).

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
