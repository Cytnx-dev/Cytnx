# Cytnx C++ test suite

## Prerequisites

GoogleTest (with GMock) must be installed and discoverable by
`find_package(GTest)` — it is a required dependency of the test suite and is
deliberately **not** fetched at configure time, so the CMake configuration stays
free of network access (see #944). Install it once:

- macOS: `brew install googletest`
- Ubuntu/Debian: `apt install libgtest-dev libgmock-dev`
- conda: `conda install -c conda-forge gtest gmock`

## Building

Build through a CMake preset that enables `RUN_TESTS` rather than a manual
`-S/-B` configure — the presets are what CI uses and keep the configuration
consistent (see `CMakePresets.json` and the top-level build docs). The `debug-*`
presets turn tests on, e.g.:

```bash
git submodule update --init --recursive        # once per fresh clone/worktree
cmake --preset debug-openblas-cpu               # macOS: debug-openblas-apple
cmake --build --preset debug-openblas-cpu --target test_main
```

Each preset builds into its own directory (`build/<preset>/`), not `build/`.

## Running

```bash
cd tests   # several suites load fixtures from tests/test_data_base
../build/<preset>/tests/test_main                        # run everything
../build/<preset>/tests/test_main --gtest_filter='BlockUniTensorTest.*'
```

or `ctest --test-dir build/<preset>` after a full build.

## macOS: do not set `DYLD_LIBRARY_PATH`

Do **not** run `test_main` with `DYLD_LIBRARY_PATH` pointed at a package
prefix such as a conda environment. `dyld` searches those directories by
*leaf name* for every library the process loads — before each library's own
install-name/rpath resolution, and case-insensitively on APFS — so Apple
system libraries get silently replaced (e.g. Accelerate's `libBLAS.dylib` by
a conda `libblas`, ImageIO codecs by conda image libraries, `libc++`, ...).
The visible symptom is a hard-to-attribute crash; the known instance is a
null-pointer call inside ARPACK's `dsaupd_` that made every
`linalg::Lanczos`/`Arnoldi` test segfault (see issue #974).

It is also unnecessary: when BLAS/LAPACK/ARPACK come from a package prefix,
CMake links them by absolute path and embeds the matching `LC_RPATH` in
`test_main`. If a binary is missing the rpath ("`Library not loaded:
@rpath/libgtest...`"), configure with
`-DCMAKE_BUILD_RPATH=<prefix>/lib`, or repair an existing binary with
`install_name_tool -add_rpath <prefix>/lib <binary>` — never with
`DYLD_LIBRARY_PATH`.
