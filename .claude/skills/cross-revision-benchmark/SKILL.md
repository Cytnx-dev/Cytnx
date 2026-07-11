---
name: cross-revision-benchmark
description: >-
  Benchmark the same Cytnx function at two or more git revisions and produce a
  fair comparison table. Use when a change is motivated by performance, a
  reviewer questions a speedup/regression, or a PR needs before/after numbers
  — e.g. comparing a branch tip against its merge base. Covers rebuilding only
  the library per revision, keeping one fixed benchmark source, Google
  Benchmark invocation, reporting rules (same machine/session, named commits,
  stated aggregation), and restoring the working tree afterwards.
---

# Cross-revision benchmarking

Performance claims are settled by measurement (never plausibility), and a
measurement is only meaningful when every column ran under identical
conditions. This skill is the procedure for benchmarking the *same* code path
at several git revisions on one machine, in one session.

## Preconditions

- The benchmark source calls a **library entry point that exists unchanged at
  every revision being compared** (e.g. `benchmarks/Tensor_bm.cpp` calls
  `Tensor` operations directly, with no dependency on the revision-specific
  internals being compared). Use the *tip's* benchmark source for every
  column — only the library underneath changes. If no existing benchmark file
  covers the function under comparison, add one first (see the `benchmarks/`
  layout) — do not write comparison code inline in a scratch harness, or the
  "one fixed source, only the library changes" guarantee this skill relies on
  is lost.
- If the function's public interface changed between revisions so the one
  benchmark source cannot compile against all of them, this procedure does not
  apply; build the benchmark per revision in separate worktrees instead.
- A **release** preset build dir is configured (`build/openblas-cpu`). Never
  benchmark a debug/ASan build.

## Which revisions

The default comparison is exactly two columns: the branch's merge base
(`git merge-base HEAD master`) and the branch tip. Add intermediate stages or
alternative implementations only when explicitly requested.

## Procedure

1. Identify the source files that differ between the revisions, with status:

   ```sh
   git diff --name-status <base> <tip> -- src include
   ```

   Split the result: files marked `M` (modified) exist at every revision and
   can be checked out directly. A file marked `A` (added by `<tip>`) does
   **not** exist at `<base>` — `git checkout <base> -- <that file>` fails with
   a pathspec error, and checking out the containing *directory* instead
   avoids the error but does not delete the file, silently leaving `<tip>`'s
   version of it in `<base>`'s build. If the diff contains any `A`/`D`
   entries, this checkout-in-place recipe does not apply to those files — fall
   back to per-revision worktrees (see Preconditions) instead of trying to
   patch around it.

2. For **each** revision, from a clean tree with only `M` files in scope:

   ```sh
   git checkout <rev> -- <modified files>       # header changes too, if any
   cmake --build build/openblas-cpu --target cytnx
   ```

   Then run the benchmark against the rebuilt library. The standalone harness
   (see `build-test-workflow`'s "Standalone harness against libcytnx.a") is
   the default path — it has no extra preconditions. The `benchmarks_main`
   CMake target is an alternative only if that build dir was already
   reconfigured with `-DRUN_BENCHMARKS=ON` (`OFF` by default in every preset)
   and has Google Benchmark installed; re-link the harness (or rebuild
   `benchmarks_main`) after each library rebuild:

   ```sh
   ./harness --benchmark_repetitions=5 --benchmark_report_aggregates_only=true \
             --benchmark_filter='<pattern>' > bm_<rev>.txt
   ```

   Checking out a changed *header* re-triggers compilation of every dependent
   translation unit — expected, let it.

3. **Restore the working tree** and rebuild back to the current revision:

   ```sh
   git checkout HEAD -- <modified files>
   cmake --build build/openblas-cpu --target cytnx
   git status --short        # must be clean
   ```

## Reporting rules

- Name the **exact commit hash** for every column; never label a column with a
  vague "before"/"old".
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
