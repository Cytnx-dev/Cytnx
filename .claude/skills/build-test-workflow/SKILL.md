---
name: build-test-workflow
description: >-
  Iterate on Cytnx builds and tests without paying for full rebuilds. Use when
  building or testing a preset, compiling after an edit, running C++
  (gtest/ctest) or Python (pytest) tests, wiring a standalone harness against
  libcytnx.a, running GPU tests when a GPU is present, or deciding how much of
  the test suite to run at each stage. Covers scripts/build_preset.sh (the one
  entry point for building and running tests), incremental builds, and
  per-suite filtering.
---

# Build & test workflow (fast iteration)

The expensive operations are configuring CMake and full rebuilds. Almost every
edit-compile-test cycle can avoid both. `CLAUDE.md` describes the presets and
the CI gates; this skill is about spending the minimum time between an edit and
its test result.

## `scripts/build_preset.sh` — the one entry point

```sh
S=.claude/skills/build-test-workflow/scripts/build_preset.sh
"$S" <preset> [--target <target>] [--test [args...]]
```

`<preset>` is any `configurePreset` name (`openblas-cpu`, `mkl-cpu`,
`debug-openblas-cpu`, …). Always build and run tests through this script
instead of composing `pip`/`cmake` by hand — it handles every detail below
correctly and gets it wrong exactly once if reimplemented ad hoc.

- **`--target <t>`** (default `all`). `pycytnx`/`all` need Python bindings
  (`BUILD_PYTHON=ON`, pybind11 discoverable); anything else (`test_main`,
  `cytnx`, `benchmarks_main`) does not and skips Python setup. A Python
  target's **first** build for a preset goes through `pip install
  --editable` (sets up a venv at `build/<preset>-venv`, makes pybind11
  discoverable, creates the import redirect pytest needs) — explicitly
  pinned to `--preset=<preset>` (pyproject.toml hardcodes
  `--preset=openblas-cpu` as its own default, so without this override every
  pip build would silently configure as `openblas-cpu` regardless of which
  preset was asked for — verified empirically). Every later call for that
  preset, Python target or not, is a direct `cmake --build` reusing the same
  build dir incrementally — no repeat pip overhead once the venv exists.
- **`--test [args]`** runs the target's tests after building.
  - Python target: args pass through to `pytest` verbatim (a path/`-k`
    filter *replaces* the default `pytests/` collection, standard pytest
    semantics — passing both would just re-add everything the filter was
    meant to narrow out); no args runs the full suite.
  - `--target benchmarks_main`: args pass through to the Google Benchmark
    binary verbatim (e.g. `--benchmark_filter=<pattern>`); no args runs
    every registered benchmark. Never run this concurrently with another
    build or benchmark — see `cross-revision-benchmark`.
  - Any other target (`test_main`, `gpu_test_main`, or both together —
    `--target "test_main gpu_test_main"`): runs through `ctest --test-dir
    <build_dir> --output-on-failure`, the same mechanism CI uses, instead of
    invoking the gtest binary directly — `test_main`/`gpu_test_main`
    register each gtest case as its own ctest test via
    `gtest_discover_tests`, so ctest gives per-test pass/fail output. A
    single optional arg becomes `-R <value>` — a ctest *regex* against
    `ClassName.TestName`, not a gtest glob/`:`-joined filter. `--test-dir`
    is used rather than `--preset`: `CMakePresets.json`'s `testPresets` are
    hardcoded to only `debug-openblas-cpu`/`debug-openblas-cuda` and don't
    generalize to every preset this script accepts. `debug-*-cuda` presets
    get `ASAN_OPTIONS` exported automatically — no need to remember the
    workaround string.
- **Max-parallelism** (`nproc`/`sysctl -n hw.ncpu`) for every build.
  `RUN_TESTS=ON` on every configure — verified to be a **zero-cost toggle**
  on an already-built dir (flipping it on a fully-built dir and re-running
  `ninja -n` showed zero pending steps), so there's no reason to default it
  off. `RUN_BENCHMARKS` stays off except for `--target benchmarks_main`:
  `CMakeLists.txt` runs `find_package(benchmark REQUIRED)` whenever it's on
  regardless of target, which would break `test_main`/`pycytnx` builds on a
  machine without Google Benchmark installed (CI's own dependency list
  doesn't include it). Same zero-cost-toggle property applies when it does
  turn on, even on an already-configured dir.
- **Generator consistency.** A build dir's generator is decided once, by
  whichever call configures it first; the script never passes a conflicting
  `-G` against an existing dir (defaulting only a *fresh* dir to Ninja, since
  that's pip/scikit-build-core's own default preference). **If a generator
  or cache mismatch does happen, never delete and reconfigure the build dir**
  — that erases the accumulated cache and forces a full rebuild. Reconfigure
  with no `-G` flag instead, so CMake keeps the dir's existing generator.

## Running tests without rebuilding

`"$S" <preset> --test [args]` (no `--target`, defaulting to `all`, or with an
explicit `--target` matching what's already built) reruns tests without
recompiling anything when the build dir is already current.

- One suite, fastest iteration signal — a direct gtest binary invocation
  skips ctest's per-test process-spawn overhead, useful when a suite has
  many cases:
  `build/debug-openblas-cpu/tests/test_main --gtest_filter='SearchTreeTest.*'`
  (glob/`:`-joined syntax, not a ctest regex — different from `--test`'s
  `-R` filter above).
- Full C++ suite via the script: `"$S" debug-openblas-cpu --target test_main --test`.
- Python: `"$S" <preset> --target pycytnx --test` (or activate the venv
  directly — `source build/<preset>-venv/bin/activate` — for a bare `pytest`
  invocation).

## How much to run, when

- **While iterating:** only the affected gtest suite / pytest file.
- **Before push / PR:** the full gates in `CLAUDE.md` — `ctest` for the CPU
  presets, `pytest pytests/` when Python-facing code changed, the CUDA
  preset compile check when GPU code changed, and (**if a GPU is actually
  present**, `nvidia-smi` succeeds) the real GPU suite too, not just the
  compile check — through the script, which builds both binaries and then
  runs every test ctest discovers for them in one `--test-dir` pass (slow,
  ~10 min):

  ```sh
  "$S" debug-openblas-cuda --target "test_main gpu_test_main" --test
  ```

  `gpu_test_main` (under `tests/gpu/`) is a separate binary from
  `test_main`, not a superset — without a visible GPU it will fail for lack
  of hardware, not because of the change under test, so the compile check
  alone (`"$S" debug-openblas-cuda`, no `--test`) is what CI expects there.
- A regression test for a bug fix must be shown to **fail on the pre-fix
  code** at least once — a guard that passes on the bug guards nothing.

## Standalone harness against libcytnx.a

For throwaway verification programs — differential checkers, fuzzers,
probes — compile a single `.cpp` directly against the built static library
instead of adding it to the build system. Keep the source in scratch/tmp,
never in the repo. Never for a benchmark of an actual Cytnx function — that
belongs as a committed `benchmarks/*.cpp` (Google Benchmark), see
`cross-revision-benchmark`.

Use a **release** preset's library (`build/openblas-cpu`); the debug library
is ASan/coverage-instrumented and a plain harness link will reject it.

```sh
g++ -std=gnu++20 -O2 -w -I include -I build/openblas-cpu \
  harness.cpp \
  build/openblas-cpu/libcytnx.a build/openblas-cpu/hptt/libhptt.a \
  -larpack -llapacke -lopenblas -lgomp \
  -o harness
```

- Order matters: harness source, then `libcytnx.a`, then `libhptt.a`, then
  the external libraries — **dependents before what they call into**
  (`arpack`/`lapacke` call BLAS/LAPACK symbols `openblas` provides).
- After rebuilding the library at a different revision, just re-run the
  same `g++` line — the harness picks up the new library.
- Linux-specific as written (`g++`, `-lgomp`); on macOS expect Homebrew
  library paths and `-lomp` in place of `-lgomp`.
