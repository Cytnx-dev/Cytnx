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

Step 1 below checks out the whole tree including `.claude/`, so `"$S"` only
survives every switch when every revision being compared already has the
script at this path — true for any comparison entirely within this repo's
history from this point forward. Comparing against a revision that predates
this script (or had it at a different path) needs it copied to a stable
location first (e.g. `cp "$S" /tmp/build_preset.sh`) before the loop starts.

Only benchmark a **committed** Google Benchmark file (`benchmarks/*.cpp`,
built as the `benchmarks_main` target). If no existing file covers the
function under comparison, add one first — do not write comparison code in a
scratch harness; a fixed, versioned benchmark source is what makes a fair
cross-revision comparison possible at all. Use a **release** preset
(`openblas-cpu` or similar) for every column — a debug/ASan build's timings
are not representative. `"$S"` strips `--coverage` instrumentation from
every build it produces (see `build-test-workflow`), so a dir it already
built for `test_main`/`pycytnx` is safe to reuse for `benchmarks_main` too.

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
2. **Case A only:** if `<rev>`'s own `benchmarks/CMakeLists.txt` doesn't
   already list `<the_bm_file>.cpp`, this revision predates the file. Copy
   just the `.cpp` from `<tip>`, then add **one line** for it to `<rev>`'s
   own `CMakeLists.txt` (an edit alongside `benchmarks_main`'s other
   sources) — do not overlay `<tip>`'s whole `CMakeLists.txt`: it may list
   other benchmark sources added or changed after `<rev>` that don't exist
   or don't compile against `<rev>`'s API, breaking the build over a
   benchmark that isn't even the one under comparison:

   ```sh
   grep -q '<the_bm_file>.cpp' benchmarks/CMakeLists.txt || {
     git checkout <tip> -- benchmarks/<the_bm_file>.cpp
     # then add "  <the_bm_file>.cpp" to CMakeLists.txt's add_executable(benchmarks_main ...) list
   }
   ```

   **Case B:** skip this — build whatever benchmark file already exists at
   this revision.
3. Build and run:

   ```sh
   "$S" openblas-cpu --target benchmarks_main --test \
     --benchmark_repetitions=5 --benchmark_report_aggregates_only=true \
     --benchmark_filter='<pattern>' > /tmp/bm-<label>.txt
   ```

4. **Case A only, and only if step 2 actually overlaid anything** (i.e.
   `<rev>` predated `<the_bm_file>.cpp`): restore before moving to the next
   revision — the working tree still holds the copied-in `.cpp` and the
   manual `CMakeLists.txt` line, so `git checkout <rev>` in the next
   iteration's step 1 would otherwise either refuse to switch or (if it can
   3-way-merge) carry that content silently forward onto the wrong revision:

   ```sh
   git restore --source=HEAD --staged --worktree -- benchmarks/<the_bm_file>.cpp
   git checkout -- benchmarks/CMakeLists.txt
   ```

   `git restore --source=HEAD` deletes `<the_bm_file>.cpp` to match `<rev>`'s
   own absence of it (a plain `checkout HEAD -- <path>` errors instead: "did
   not match any file(s) known to git" for a path HEAD doesn't have).
   `CMakeLists.txt` itself is a plain modification (the added line), not a
   new path, so `git checkout -- <path>` reverts it correctly on its own.

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
