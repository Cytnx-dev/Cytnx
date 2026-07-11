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
  every revision being compared** (e.g. `benchmarks/search_tree_bm.cpp` calls
  `cytnx::OptimalTreeSolver::solve` directly). Use the *tip's* benchmark source
  for every column — only the library underneath changes.
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

1. Identify the source files that differ between the revisions:

   ```sh
   git diff --name-only <base> <tip> -- src include
   ```

2. For **each** revision, from a clean tree:

   ```sh
   git checkout <rev> -- <those files>          # header changes too, if any
   cmake --build build/openblas-cpu --target cytnx   # or: ninja libcytnx.a in the dir
   ```

   Then run the benchmark against the rebuilt library — either the
   `benchmarks_main` target, or a standalone harness link (see the
   `build-test-workflow` skill) rebuilt after each `ninja`:

   ```sh
   ./bench --benchmark_repetitions=5 --benchmark_report_aggregates_only=true \
           --benchmark_filter='<pattern>' > bm_<rev>.txt
   ```

   Checking out a changed *header* re-triggers compilation of every dependent
   translation unit — expected, let it.

3. **Restore the working tree** and rebuild back to the current revision:

   ```sh
   git checkout HEAD -- <those files>
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
