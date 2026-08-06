# Contributing to Cytnx

Thanks for considering a contribution. This guide covers the development
environment and the housekeeping that is easy to miss when a change touches
metadata shared across the build, packaging, and docs. Maintainers cutting a
tagged release should follow [RELEASING.md](RELEASING.md) instead.

## Development environment

[Pixi](https://pixi.sh) provisions everything cytnx builds against from
conda-forge — the toolchain, the native libraries, and the Python interpreter —
locked to identical versions on Linux, macOS, and Windows. `pixi.toml` is the
list; it is not repeated here. Nothing else needs to be installed system-wide,
with one exception: on Windows, MSVC itself must already be present.

Install Pixi, then from the repository root:

```sh
pixi config set --local detached-environments true   # see the note below
pixi install          # materialize the environment from pixi.lock
pixi run doctor       # report the toolchain versions Pixi resolved
pixi run test-cpp     # configure, build test_main, run the C++ suite
```

`pixi run` initializes the git submodules on first use.

### Choosing a preset

There is one environment, and every task takes the name of a preset from
`CMakePresets.json` as its first argument. Nothing about the environment fixes
which preset is used, so consecutive commands can build different ones:

```sh
pixi run test-cpp                     # openblas-cpu
pixi run test-cpp debug-mkl-cpu       # ASan + MKL, no environment switch
pixi run build openblas-cuda          # compile-check CUDA; no GPU required
pixi run bench mkl-cpu Contract       # one benchmark filter
```

The default is `openblas-cpu` — the preset `pyproject.toml` pins for the PyPI
wheel — so the everyday build matches what is released. On Windows the default
is `mkl-cpu` instead, because that is what the conda package is built with on
x86 and there is no Windows wheel. Both BLAS vendors are installed on x86, so
switching between `openblas-*` and `mkl-*` costs nothing; `linux-aarch64` and
`osx-arm64` have no MKL and so no `mkl-*` build.

The CUDA toolkit comes from NVIDIA's PyPI wheels, the same ones the release
build installs — the version ranges live in `tools/prepare_cuda_release.py` and
`pixi.toml` follows them — so a local CUDA build and a released `cytnx-cuda`
wheel compile against one toolchain. It is installed on every platform NVIDIA
publishes for, because compiling a CUDA preset needs no GPU; only *running* GPU
code does. Linux gets cuTensorNet and cuStateVec and so builds the ordinary
`*-cuda` presets, while NVIDIA publishes neither for Windows (#1111), where the
`*-cuda-windows` presets turn `USE_CUQUANTUM` off. macOS has no CUDA at all.

Every task works in the one build tree its preset names, `build/<preset>/`, so
the C++ suite and the Python tests share a single set of object files, an
incremental rebuild serves both, and a Pixi build is the same tree as a manual
`cmake --preset` build. `clean` reclaims one tree without touching the others.

| task | what it does |
|---|---|
| `setup` | initialize the git submodules |
| `doctor` | print the resolved compiler, CMake, Ninja, and Python versions |
| `configure` / `build` / `test-cpp` | the C++ build and GoogleTest suite |
| `install-python` / `test-python` | editable install of the extension, then pytest |
| `gate` | the pre-pull-request run: `test-cpp` under both `debug-*` CPU presets |
| `bench` | build and run `benchmarks_main`, optionally filtered |
| `stubs` / `stubtest` | regenerate the committed type stubs, and check them |
| `clean` | remove one preset's build tree |
| `format` | run the pre-commit hooks over the whole tree |

`install-python` installs `.[dev]`, so the Python packages come from
`pyproject.toml` rather than being declared a second time in `pixi.toml`. Only
one preset's extension is importable at a time; re-running the task for another
preset is cheap, because that preset's build tree is already there.

Before opening a pull request, run `pixi run gate` on an x86 machine: it builds
and runs the C++ suite under `debug-openblas-cpu` and `debug-mkl-cpu`, both
with AddressSanitizer, which is what CI gates on. On `linux-aarch64` and
`osx-arm64` there is no MKL, so `pixi run test-cpp debug-openblas-cpu` is the
whole local gate and CI covers the MKL half.

`test-python` against a `debug-*` preset works on Linux only:
`tools/run_pytest.py` preloads the sanitizer runtime that an instrumented
extension needs before an ordinary interpreter can import it, and `LD_PRELOAD`
is a Linux mechanism. macOS would need `DYLD_INSERT_LIBRARIES` aimed at Clang's
ASan dylib, which cytnx does not wire up; run the Python tests against a
release preset there.

### On Windows

Install Visual Studio 2022 or Visual Studio 2022 Build Tools with the "Desktop
development with C++" workload, MSVC v143 x64/x86 build tools, and a Windows 10
or 11 SDK; Pixi cannot supply the compiler itself. Run the tasks from an x64
PowerShell prompt at the repository root.

Two dependencies ship layouts MSVC cannot consume directly, so
`tools/prepare_windows_import_libraries.py` derives what is missing from the
installed files — it vendors nothing and is idempotent. The configure tasks
depend on it; `pixi run check-windows-layout` inspects without changing
anything. The remaining Windows-specific settings are commented where they are
declared, in `pixi.toml`.

### The detached-environments setting

Pixi's default layout puts the environment in `.pixi/` inside the checkout, and
cytnx adds its LAPACKE (and, in CUDA builds, cuTENSOR) include directory to
`cytnx`'s `PUBLIC` usage requirements. CMake refuses to generate when an
exported include directory lies inside the source tree, since that path would
be baked into `CytnxTargets.cmake`:

```
Target "cytnx" INTERFACE_INCLUDE_DIRECTORIES property contains path:
  ".../\.pixi/envs/default/include" which is prefixed in the source directory.
```

`detached-environments` moves the environment outside the checkout and the
generate step succeeds. #1120 tracks the export contract itself; once cytnx
stops exporting dependency include directories as build-tree paths, the setting
is no longer needed.

### Changing dependencies

Edit `pixi.toml` and commit the regenerated `pixi.lock` alongside it. `pixi
lock` resolves every platform in the workspace at once, so the lock has to be
regenerated even for a change that affects only one of them.

## Updating the minimum supported Python version

The minimum Python version is declared independently in a few places, since
none of them can import it from a single source:

- **`pyproject.toml`** — `requires-python` under `[project]`. This is the
  version cibuildwheel reads to decide which interpreters to build PyPI
  wheels for.
- **`conda_build/conda_build_config.yaml`** — the `python:` list. This file
  is plain YAML and is read by conda-build *before* `meta.yaml` is rendered,
  so it cannot import the bound from `pyproject.toml` the way `meta.yaml`
  does for the package version.
- **`docs/source/adv_install.rst`** — the `python >= X.Y` requirement.

When raising (or lowering) the minimum, update all three. There is no
automated check for this — please grep for the old version string (e.g.
`3.9`) across the repository before opening the PR to catch anything missed.

## Regenerating the Python type stubs

The `cytnx` Python package ships PEP 561 type stubs (`cytnx/cytnx/*.pyi`)
committed to the repository and shipped unchanged in every wheel and conda
package. They are generated from the built `cytnx.cytnx` pybind11 extension
by `tools/generate_stubs.py`, not written by hand.

**If your change touches any `pybind/*.cpp` binding** — a new function, a
changed signature, a different default value, a different overload set — the
committed stubs go stale and must be regenerated as part of the same PR.

Stub generation is only reproducible when the tools that produce it are
pinned, since both silently change the emitted annotations across versions:

- `pybind11` (`[build-system].requires` in `pyproject.toml`) — controls the
  type annotations baked into the compiled extension.
- `pybind11-stubgen` (`dev` extra in `pyproject.toml`) — walks the built
  extension and renders the `.pyi` files.

Both are pinned to exact versions in `pyproject.toml`, where a comment beside
each pin reminds you to regenerate the committed stubs whenever it is bumped.
The `requires-python` floor matters too, since the stubs are generated with
the lowest supported interpreter (see below).

To regenerate:

1. Build the extension and install the pinned dev tools together, through
   the editable install. Go through `pip` rather than a direct `cmake`
   configure/build: the `pip` path provisions the pinned `pybind11` from
   `[build-system].requires` via build isolation, so the extension — and thus
   the regenerated stubs — are built against exactly that version. (A direct
   `cmake` build instead uses whatever compatible `pybind11` is already
   installed.)
   ```sh
   pip install --editable '.[dev]'
   ```
2. Regenerate the committed stubs:
   ```sh
   python tools/generate_stubs.py
   ```
   The generator introspects the installed `cytnx.cytnx` (the editable install
   from step 1), falling back to a build under `build/`; pass `--extension` to
   point at a specific `.so`/`.pyd` to override both. Run this with the lowest
   supported interpreter (the `requires-python` floor declared in
   `pyproject.toml`) so the emitted syntax stays parseable everywhere the
   package is installed.
3. Review the diff under `cytnx/cytnx/*.pyi` and commit it alongside the
   binding change that caused it.

`mypy.stubtest` compares the committed stubs against the live runtime module
and catches mismatches (missing members, incompatible defaults, overloads
that can never match). It is not yet wired into CI, so run it manually after
regenerating:

```sh
python -m mypy.stubtest cytnx.cytnx
```
