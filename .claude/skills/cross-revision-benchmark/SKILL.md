---
name: cross-revision-benchmark
description: >-
  Benchmark the same Cytnx function at two or more git revisions and produce a
  fair comparison table. Use when a change is motivated by performance, a
  reviewer questions a speedup/regression, or a PR needs before/after numbers
  — e.g. comparing a branch tip against its merge base. Covers per-revision
  git worktrees, the fixed-benchmark-source vs per-revision-benchmark-source
  cases, when to use a committed Google Benchmark file vs a throwaway
  standalone harness, not running benchmarks concurrently, and reporting
  rules (same machine/session, commit hash plus a rebase-proof readable
  label, stated aggregation).
---

# Cross-revision benchmarking

Performance claims are settled by measurement (never plausibility), and a
measurement is only meaningful when every column ran under identical
conditions. This skill is the procedure for benchmarking the *same* code path
at several git revisions on one machine, in one session.

## Which of two cases applies

- **Case A — the benchmark's call signature is unchanged across every
  revision being compared** (e.g. `benchmarks/Tensor_bm.cpp` calls `Tensor`
  operations whose signature is stable). Keep **one fixed benchmark source**
  (the tip's) and vary only the library underneath.
- **Case B — the function's public interface changed** between revisions, so
  one benchmark source cannot compile against all of them (or a revision
  being compared predates the benchmark file existing at all — e.g. the
  merge-base commit in the history that originally motivated this skill,
  #853/#1027, had no `search_tree_bm.cpp` yet). Use **each revision's own
  benchmark source** instead of forcing one file across all columns.

Both cases share the same underlying mechanism (below): a full `git checkout
<rev>`, verified to be an **incremental, not a full, rebuild** even for a
non-trivial change (tested empirically on the actual #853/#1027 history:
switching between two commits with a 5-file, 460-line diff — including one
newly-added file — in an already-built dir left only 7 build steps
pending: the touched translation units, their link-time dependents, and
nothing else. Git only rewrites files whose blob content actually differs
between the two commits, so unrelated files keep their old mtime and
Ninja/Make correctly leaves them alone). This means whole-tree checkout is
cheap enough to be the default, and it is the only form that behaves
correctly on renamed/added/removed files — no pathspec errors, no
directory-checkout-doesn't-delete trap.

If no existing benchmark file covers the function under comparison at all,
add one first (see the `benchmarks/` layout) rather than writing comparison
code inline in a scratch harness — a committed Google Benchmark file is what
makes both cases above possible in the first place.

A **release** preset build dir is used for every column (`build/openblas-cpu`
or a fresh worktree of it). Never benchmark a debug/ASan build.

## Which revisions

The default comparison is exactly two columns: the branch's merge base
(`git merge-base HEAD master`) and the branch tip. Add intermediate stages or
alternative implementations only when explicitly requested.

## Procedure

**Prefer a git worktree per revision** (`git worktree add /tmp/bm-<label>
<rev>`) over checking out revisions in place in the main working tree. A
worktree never touches the current branch, needs no "restore at the end"
step, and lets independent revisions' builds run **in parallel** (each has
its own build dir, so there is no shared-cache race) — genuinely useful
since a from-scratch worktree build has no prior cache to build
incrementally from, unlike reusing one tree's build dir across sequential
checkouts. Kick off all the builds, then benchmark **each as soon as its own
build finishes** rather than waiting for the slowest one — but never run two
benchmarks at the same time (see Reporting rules): building may parallelize
across revisions, running a benchmark must not.

1. For each revision, create a worktree and configure a release build:

   ```sh
   git worktree add /tmp/bm-<label> <rev>
   cd /tmp/bm-<label>
   cmake --preset openblas-cpu -G Ninja -DBUILD_PYTHON=OFF \
     -DRUN_TESTS=ON -DRUN_BENCHMARKS=ON
   ```

2. **Case A (fixed benchmark source):** after configuring, overlay the tip's
   benchmark file **and its build registration** on top of this revision's
   tree before building — `benchmarks/CMakeLists.txt` lists every `.cpp` in
   the target's `add_executable(...)` sources, so overlaying only the `.cpp`
   silently leaves it uncompiled if this revision predates the file existing
   at all (verified by testing exactly this: building the merge-base commit
   from #853/#1027 — which never had `search_tree_bm.cpp` — with only the
   `.cpp` overlaid produced a binary with the file simply missing; adding
   `benchmarks/CMakeLists.txt` to the overlay fixed it and produced a real
   result). Overlay both:

   ```sh
   git -C /tmp/bm-<label> checkout <tip> -- \
     benchmarks/<the_bm_file>.cpp benchmarks/CMakeLists.txt
   ```

   **Case B (per-revision benchmark source):** skip this step — build
   whatever benchmark file already exists in that revision's own checkout.

3. Build and run, once per worktree (in parallel across worktrees is fine for
   this step; never for the run below):

   ```sh
   cmake --build build/openblas-cpu --target benchmarks_main --parallel "$(nproc)"
   ```

   Use the **committed** `benchmarks_main` target (Google Benchmark, built
   via CMake) for any benchmark of an actual Cytnx function — that is what
   the `benchmarks/` directory and this whole procedure exist for. Reach for
   a standalone, uncommitted harness (see `build-test-workflow`'s "Standalone
   harness against libcytnx.a") only for a **language/stdlib feature
   question that has nothing to do with Cytnx's own code** — e.g. comparing
   raw `std::vector` vs `std::unordered_set` operation cost to decide which
   to use in an implementation. That kind of question does not belong as a
   permanent `benchmarks/*.cpp` file, and it does not need cross-revision
   comparison at all (the answer does not depend on which Cytnx commit is
   checked out) — it is a one-off, run and discarded.

   Run each worktree's binary **one at a time**, never concurrently with
   another benchmark run — CPU contention between simultaneous runs skews
   both:

   ```sh
   ./build/openblas-cpu/benchmarks/benchmarks_main \
     --benchmark_repetitions=5 --benchmark_report_aggregates_only=true \
     --benchmark_filter='<pattern>' > /tmp/bm-<label>.txt
   ```

4. Remove the worktrees once all columns are recorded:

   ```sh
   git worktree remove /tmp/bm-<label>
   ```

   (The alternative — checking out revisions sequentially in the main
   working tree's own `build/openblas-cpu`, restoring to `HEAD` afterwards —
   is cheaper in total CPU time for Case A, since it reuses one incremental
   build across columns instead of N fresh ones, at the cost of losing the
   parallel-build option and briefly leaving the main tree on another
   commit. Prefer worktrees; fall back to this only when the machine cannot
   afford N build dirs' disk space.)

## Reporting rules

- Every column needs **both** the exact commit hash **and** a short, stable,
  readable label ("master", "merge-base", "min-heap fix") — a hash alone goes
  stale the moment history is rewritten (rebase, squash, force-push changes
  what commit that label used to point at), while the label is what the
  comparison is actually *about* and survives that. State both together,
  e.g. "min-heap fix (`5d4ca56`)", never one without the other.
- All columns fresh in the **same session, same machine, same build flags** —
  never mix numbers from a previous run, another machine, or a different
  preset.
- State the aggregation ("mean of 5 repetitions") and the unit.
- Compare against a **correct** baseline. If an old revision is buggy in a way
  that changes how much work it does (e.g. out-of-bounds reads causing skipped
  work), its numbers are not a correctness-preserving baseline — say so
  explicitly rather than letting the table imply a fair race.
- Record the deciding numbers in the PR description or commit message so the
  performance choice is auditable.
