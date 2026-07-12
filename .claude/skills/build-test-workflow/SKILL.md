---
name: build-test-workflow
description: >-
  Build and test Cytnx through scripts/build_preset.sh — the one entry point
  for building any CMake preset and running C++ (gtest/ctest) or Python
  (pytest) tests. Use when building a preset, compiling after an edit,
  running tests, running GPU tests when a GPU is present, or wiring a
  standalone harness against libcytnx.a.
---

# Build & test workflow

## `scripts/build_preset.sh` — the one entry point

```sh
S=.claude/skills/build-test-workflow/scripts/build_preset.sh
"$S" <preset> [--target <target>] [--test [args...]]
```

Always build and test through this script instead of composing `pip`/`cmake`
by hand.

- **`<preset>`** — any `configurePreset` in `CMakePresets.json`:
  `openblas-cpu`, `mkl-cpu`, `openblas-cuda`, `mkl-cuda`, `openblas-apple`,
  or their `debug-*` variants (Debug + ASan + `RUN_TESTS=ON`). Iterate on a
  `debug-*-cpu` preset; benchmarks need a release preset.
- **`--target <t>`** (default `all`) — `all`/`pycytnx` (Python bindings),
  `test_main`, `gpu_test_main`, `benchmarks_main`, `cytnx`, or a
  space-separated list (`"test_main gpu_test_main"`).

**Build logic.** A Python target's (`all`/`pycytnx`) first build for a
preset runs `pip install --editable` into `build/<preset>-venv` — this is
what installs pybind11 and the other build requirements and creates the
import redirect pytest needs. Every later call, Python target or not, is a
plain incremental `cmake --build` of the same `build/<preset>` dir — no
repeat pip overhead.

**Never delete a build dir over a generator or cache mismatch** — that
erases the accumulated cache and forces a full rebuild. Reconfigure with no
`-G` flag instead, so CMake keeps the dir's existing generator.

**`--test [args]`** runs the built target's tests; args depend on the
target:

- Python target: args pass to `pytest` verbatim and replace the default
  collection; no args runs `pytest pytests/ --doctest-modules`.
- `benchmarks_main`: args pass to the Google Benchmark binary verbatim
  (`--benchmark_filter=<pattern>`, …).
- `test_main`/`gpu_test_main`: runs ctest scoped to the requested binary's
  tests; a single optional arg becomes `-R <value>` — a ctest *regex*
  against `ClassName.TestName`, not a gtest glob. For the fastest
  single-suite signal skip ctest's per-test overhead and invoke the binary
  directly:
  `build/debug-openblas-cpu/tests/test_main --gtest_filter='Storage.*'`
  (gtest glob/`:` syntax).

## GPU rules

- **No GPU** (`nvidia-smi` fails): compile check only — `"$S"
  debug-openblas-cuda` with no `--test`. Building `gpu_test_main` never
  needs a GPU.
- **GPU present**: also run the real suite:
  `"$S" debug-openblas-cuda --target "test_main gpu_test_main" --test`.
- `debug-*-cuda` presets need
  `ASAN_OPTIONS='protect_shadow_gap=0:replace_intrin=0:detect_leaks=0'` to
  run at all; the script exports it automatically.

## How much to run, when

- **While iterating:** only the affected gtest suite / pytest file.
- **Before push / PR:** the full gates in `CLAUDE.md` — ctest for both CPU
  debug presets, `pytest pytests/` when Python-facing code changed, the
  CUDA compile check when GPU code changed, and the GPU suite when a GPU is
  present.

## Regression discipline

A regression test for a bug fix must be shown to fail on the pre-fix code
at least once — a guard that passes on the bug guards nothing.

## Standalone harness against libcytnx.a

For throwaway verification programs (differential checkers, probes):
compile one `.cpp` directly against the built static library; keep the
source in scratch/tmp, never in the repo. Never for a benchmark of a Cytnx
function — that belongs in `benchmarks/` (see `cross-revision-benchmark`).
Use a release preset's library — a debug preset's is ASan-instrumented and
won't link plainly.

```sh
g++ -std=gnu++20 -O2 -w -I include -I build/openblas-cpu \
  harness.cpp \
  build/openblas-cpu/libcytnx.a build/openblas-cpu/hptt/libhptt.a \
  -larpack -llapacke -lopenblas -lgomp \
  -o harness
```

Link order matters: dependents before what they call into. Linux-specific
as written; on macOS expect Homebrew library paths and `-lomp` in place of
`-lgomp`.
