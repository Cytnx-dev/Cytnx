---
name: build-test-workflow
description: >-
  Iterate on Cytnx builds and tests without paying for full rebuilds. Use when
  building a preset (via tools/build_preset.sh), compiling after an edit,
  running C++ (gtest/ctest) or Python (pytest) tests during development,
  wiring a quick standalone harness against the built library, running GPU
  tests when a GPU is present, or deciding how much of the test suite to run
  at each stage. Covers per-preset venvs, generator-consistent pip/cmake
  builds, incremental/accumulated builds, running ctest and pytest without
  reconfiguring or rebuilding, per-suite gtest filtering, ASan invocation
  (debug+CUDA presets), and the exact link line for compiling a scratch .cpp
  against libcytnx.a.
---

# Build & test workflow (fast iteration)

The expensive operations are configuring CMake and full rebuilds. Almost every
edit-compile-test cycle can avoid both. `CLAUDE.md` describes the presets and
the CI gates; this skill is about spending the minimum time between an edit and
its test result.

## Building a preset: use tools/build_preset.sh

```sh
tools/build_preset.sh <preset> --pytest       # venv + pip editable install; enables pytest
tools/build_preset.sh <preset> --gtest-only   # direct cmake configure+build; no venv, no wheel
```

`<preset>` is any `configurePreset` name (`openblas-cpu`, `mkl-cpu`,
`debug-openblas-cpu`, …). This is the default entry point for building —
prefer it over composing `pip install`/`cmake` by hand, since it already
handles the details below correctly:

- **A dedicated venv per preset**, at `build/<preset>-venv` (under the
  already-gitignored `build/`, so no new venv ever needs manual cleanup or a
  `.gitignore` entry). `--pytest` creates it on first use and reuses it after;
  `pip install --editable` is idempotent, so re-running `--pytest` to pick up
  new C++ changes is safe and cheap on a warm venv.
- **Max-parallelism builds** (`nproc` on Linux, `sysctl -n hw.ncpu` on macOS)
  for both the pip path and direct `cmake --build`.
- **A consistent generator between `pip install` and `cmake`.** A build dir's
  generator is decided once, by whichever entry point configures it *first* —
  the script never passes a conflicting `-G` against an already-configured
  dir, only defaulting a brand-new dir to Ninja (matching
  pip/scikit-build-core's own default preference, so both entry points
  converge on the same generator without needing to agree in advance).
- `RUN_TESTS=ON` and `RUN_BENCHMARKS=ON` on every build the script produces.
  Both were verified to be a **zero-cost toggle** on an already-built dir —
  flipping either on a fully-built `build/openblas-cpu` and re-running `ninja
  -n` showed zero pending build steps in either direction — so there is no
  reason to default them off and pay a later reconfigure+rebuild to turn them
  on.

**When a generator or cache mismatch occurs, do not delete and reconfigure
the build dir.** That erases the accumulated cache and forces a full rebuild.
Mixing `pip install -e` (Ninja by default) and a raw `cmake --preset` (the
platform default — typically Unix Makefiles, even with Ninja installed, since
no preset sets `generator`) is the usual cause. Recovery is to build a preset
with **`--pytest` first** (whenever `pytest`-driven testing might be needed
later) so the dir's generator is Ninja from the start; if a mismatch does
happen, reconfigure with **no `-G` flag** so CMake keeps the dir's existing
generator, and never blindly delete a working build dir to "fix" this.

## Accumulated (incremental) builds

- **Never delete `build/` or any `build/<preset>/`.** The CMake cache, object
  files, and ccache entries are what make the next build take seconds instead
  of an hour.
- After an edit, rebuild only the target you need in the already-configured
  build dir:

  ```sh
  cmake --build build/debug-openblas-cpu --target test_main   # C++ tests
  cmake --build build/openblas-cpu --target cytnx              # release lib
  ```

  This is generator-agnostic and always works — the library target's actual
  name is `cytnx`, not `libcytnx.a`, so a Make-vs-Ninja-specific invocation
  isn't even a correct shortcut. Prefer this or `tools/build_preset.sh` over
  running the generator directly.
- For the Python editable install, `tools/build_preset.sh <preset> --pytest`
  is only needed once per preset (or when packaging/config changes). After
  C++ edits, rebuild just the extension target in that preset's build dir:

  ```sh
  cmake --build build/<preset> --target pycytnx
  ```

  `pycytnx` is the pybind11 module target; building the default target wastes
  time on everything else.

## Running tests without rebuilding

`ctest` and `pytest` do not compile anything — they run what already exists.

- **C++, one suite (the iteration loop):** run the gtest binary directly with a
  filter; this is the fastest signal after a rebuild of `test_main`:

  ```sh
  build/debug-openblas-cpu/tests/test_main --gtest_filter='SearchTreeTest.*'
  ```

  Debug presets build with AddressSanitizer. This is only needed for the
  **debug + CUDA** presets (`debug-openblas-cuda`, `debug-mkl-cuda`) — if a
  `debug-openblas-cuda`/`debug-mkl-cuda` binary aborts inside ASan on
  startup, prefix with
  `ASAN_OPTIONS='protect_shadow_gap=0:replace_intrin=0:detect_leaks=0'`.
- **C++, full suite:** `ctest --preset cpu-only --output-on-failure`, or for a
  build dir without a ctest preset,
  `ctest --test-dir build/debug-mkl-cpu --output-on-failure`.
- **Python, one preset's venv:** activate that preset's venv (built via
  `tools/build_preset.sh <preset> --pytest`) before running pytest, so the
  right compiled extension is on `sys.path`:

  ```sh
  source build/openblas-cpu-venv/bin/activate
  pytest pytests/ # or a single file/test node
  ```

  Rebuild `pycytnx` first only if C++ code changed (see above); pure-Python
  edits under `cytnx/` or `pytests/` need no build step at all. Deactivate
  (or open a new shell) before switching to a different preset's venv.

## How much to run, when

- **While iterating:** only the affected gtest suite via `--gtest_filter` (and
  the affected pytest file). Everything else is noise until the change settles.
- **Before push / PR:** the full gates in `CLAUDE.md` — `ctest` for the CPU
  presets, `pytest pytests/` when Python-facing code changed, and the CUDA
  preset compile check when GPU code changed.
- **If a GPU is actually present** (`nvidia-smi` succeeds), don't stop at the
  CUDA compile check when GPU code changed — also run the GPU test suite:

  ```sh
  cmake --preset debug-openblas-cuda
  cmake --build --preset debug-openblas-cuda --target gpu_test_main test_main
  ctest --preset cpu-and-cuda --output-on-failure   # slow (~10 min)
  ```

  `gpu_test_main` (under `tests/gpu/`) is the GPU suite's binary — a separate
  target from `test_main`, not a superset of it. Without a visible GPU, the
  compile check from `CLAUDE.md` is the only thing to run; do not attempt
  `gpu_test_main` there, it will fail for lack of hardware, not because of
  the change under test.
- A regression test for a bug fix must be shown to **fail on the pre-fix
  code** at least once (rebuild the lib at the old revision, run the new test,
  watch it fail) — a guard that passes on the bug guards nothing.

## Standalone harness against libcytnx.a

For throwaway verification programs — differential checkers, fuzzers, probes,
micro-benchmarks — compile a single `.cpp` directly against the built static
library instead of adding targets to the build system. Keep the source in the
scratchpad/tmp area, never in the repo.

Use a **release** preset's library (`build/openblas-cpu`); the debug library
contains ASan/coverage-instrumented objects that a plain harness link will
reject.

```sh
g++ -std=gnu++20 -O2 -w -I include -I build/openblas-cpu \
  harness.cpp \
  build/openblas-cpu/libcytnx.a build/openblas-cpu/hptt/libhptt.a \
  -larpack -llapacke -lopenblas -lgomp \
  -o harness
```

Notes:

- Order matters: the harness source first, then `libcytnx.a`, then `libhptt.a`,
  then the external libraries, **dependents before the libraries they call
  into** — `arpack` and `lapacke` call BLAS/LAPACK symbols that `openblas`
  provides, so they must precede `-lopenblas` on the command line.
- `-I build/<preset>` is harmless today (the only `configure_file` in the
  build is the Doxyfile, so no build dir currently has a generated header to
  find) — kept in case that changes; `-w` silences warnings from public
  headers that the harness cannot fix.
- For a Google-Benchmark harness, append
  `/usr/lib/x86_64-linux-gnu/libbenchmark_main.a -lbenchmark -lpthread`
  (the `_main` archive provides `main()`). Building the `benchmarks_main`
  CMake target instead needs a reconfigure with `-DRUN_BENCHMARKS=ON`
  (`OFF` by default in every preset unless built via `tools/build_preset.sh`,
  which always turns it on — verified a zero-cost toggle, see above) and
  requires Google Benchmark installed (`find_package(benchmark REQUIRED)`) —
  the standalone harness has no such precondition, which is why it is the
  default path here.
- After rebuilding the library at a different revision (`cmake --build
  build/openblas-cpu --target cytnx`), just re-run the same `g++` line — the
  harness picks up the new library.
- Linux-specific as written (`g++`, `-lgomp`,
  `/usr/lib/x86_64-linux-gnu/libbenchmark_main.a`); on macOS expect Homebrew
  library paths and `-lomp` in place of `-lgomp`.
