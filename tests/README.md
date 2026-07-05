# Cytnx C++ test suite

## Building

```bash
git submodule update --init --recursive   # required once per fresh clone/worktree
cmake -S . -B build -DRUN_TESTS=ON
cmake --build build --target test_main
```

GoogleTest is found via `find_package(GTest)` if installed, and otherwise
fetched automatically as a pinned release, so no system googletest is
required.

## Running

```bash
cd tests   # several suites load fixtures from tests/test_data_base
../build/tests/test_main                      # run everything
../build/tests/test_main --gtest_filter='BlockUniTensorTest.*'
```

or `ctest --test-dir build` after a full build.

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
