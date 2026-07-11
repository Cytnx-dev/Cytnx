---
name: build-test-workflow
description: >-
  Iterate on Cytnx builds and tests without paying for full rebuilds. Use when
  compiling after an edit, running C++ (gtest/ctest) or Python (pytest) tests
  during development, wiring a quick standalone harness against the built
  library, or deciding how much of the test suite to run at each stage. Covers
  incremental/accumulated builds, running ctest and pytest without
  reconfiguring or rebuilding, per-suite gtest filtering, ASan invocation, and
  the exact link line for compiling a scratch .cpp against libcytnx.a.
---

# Build & test workflow (fast iteration)

The expensive operations are configuring CMake and full rebuilds. Almost every
edit-compile-test cycle can avoid both. `CLAUDE.md` describes the presets and
the CI gates; this skill is about spending the minimum time between an edit and
its test result.

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

  This is generator-agnostic and always works. No preset sets a `generator`,
  so a plain `cmake --preset <name>` configure picks the platform default —
  typically Unix Makefiles, not Ninja, even where Ninja is installed. Only run
  the generator directly (`cd build/openblas-cpu && ninja libcytnx.a`) if you
  know that specific build dir was configured with `-G Ninja` (check
  `CMAKE_GENERATOR:INTERNAL` in its `CMakeCache.txt`); the library target's
  actual name is `cytnx`, not `libcytnx.a`, so the Make and Ninja spellings
  differ too, and the `cmake --build --target cytnx` form is the one safe to
  assume everywhere.
- Re-run `cmake --preset <name>` only when CMakeLists/preset files changed.
  If a build dir errors about a generator or cache mismatch (usually from
  mixing configure methods), delete only that one preset dir and reconfigure —
  not all of `build/`.
- For the Python editable install, the `pip install --editable` step is only
  needed once (or when packaging/config changes). After C++ edits, rebuild the
  extension target in the build dir the editable install was configured with:

  ```sh
  cmake --build <python-build-dir> --target pycytnx
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

  Debug presets build with AddressSanitizer; if the binary aborts inside ASan
  on startup, prefix with
  `ASAN_OPTIONS='protect_shadow_gap=0:replace_intrin=0:detect_leaks=0'`.
- **C++, full suite:** `ctest --preset cpu-only --output-on-failure`, or for a
  build dir without a ctest preset,
  `ctest --test-dir build/debug-mkl-cpu --output-on-failure`.
- **Python:** `pytest pytests/` (or a single file/test node) runs against the
  already-built extension. Rebuild `pycytnx` first only if C++ code changed;
  pure-Python edits under `cytnx/` or `pytests/` need no build step at all.

## How much to run, when

- **While iterating:** only the affected gtest suite via `--gtest_filter` (and
  the affected pytest file). Everything else is noise until the change settles.
- **Before push / PR:** the full gates in `CLAUDE.md` — `ctest` for the CPU
  presets, `pytest pytests/` when Python-facing code changed, and the CUDA
  preset compile check when GPU code changed.
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
  CMake target instead needs a one-time reconfigure with
  `-DRUN_BENCHMARKS=ON` (`OFF` by default in every preset) and requires
  Google Benchmark installed (`find_package(benchmark REQUIRED)`) — the
  standalone harness has no such precondition, which is why it is the
  default path here.
- After rebuilding the library at a different revision (`cmake --build
  build/openblas-cpu --target cytnx`), just re-run the same `g++` line — the
  harness picks up the new library.
- Linux-specific as written (`g++`, `-lgomp`,
  `/usr/lib/x86_64-linux-gnu/libbenchmark_main.a`); on macOS expect Homebrew
  library paths and `-lomp` in place of `-lgomp`.
