# Developing Cytnx on Windows

Windows uses the same Pixi environments and tasks as Linux and macOS —
see [Development environment](../../CONTRIBUTING.md#development-environment)
for those. This page covers what is specific to Windows.

## Prerequisites

Pixi cannot supply MSVC, so install Visual Studio 2022 or Visual Studio 2022
Build Tools first, with:

- Desktop development with C++
- MSVC v143 x64/x86 build tools
- A Windows 10 or Windows 11 SDK

Pixi's `c-compiler` and `cxx-compiler` packages locate and activate that
installation; they do not contain the compiler. Run the Pixi tasks from an x64
PowerShell prompt at the repository root — each one re-enters
`tools/activate_windows.bat`, which resolves the Visual Studio installation
with `vswhere` and puts MSVC, the Pixi environment, and any CUDA directories on
`PATH` inside a single process. CMake drives `cl.exe` through the Ninja
generator.

The tasks use the MKL environment, so the CPU build is
`pixi run -e mkl test-cpp`.

One option is forced off on Windows and nowhere else:
`CMAKE_INTERPROCEDURAL_OPTIMIZATION`, because MSVC's `/GL` rejects the mixed
static-archive link cytnx produces while `CMakeLists.txt` otherwise enables it
on every non-Apple platform.

## Dependency layouts that need repair

Two of the packages Windows resolves ship a layout CMake cannot consume
directly, so `tools/prepare_windows_import_libraries.py` derives what is
missing from the installed files. It vendors nothing and is idempotent; the
configure tasks depend on it, and it can also be run on its own:

```powershell
pixi run -e mkl check-windows-layout          # inspect without invoking lib.exe
pixi run -e mkl prepare-windows
```

- conda-forge's ARPACK ships a MinGW DLL with a GNU import archive, so the
  script reads the DLL's PE export table and rebuilds an MSVC `.lib` with
  `lib.exe`.
- NVIDIA's CUDA 13 math-library wheels omit their MSVC import libraries, and
  the NVVM wheel installs its DLL under `bin/x86_64` while CUDA 13's nvlink
  searches `bin/x64`. The `--cuda` mode generates the missing import libraries
  and provides the canonical NVVM path.

## CUDA

CUDA is an optional Pixi environment rather than a base dependency. It sources
nvcc, CCCL, the runtime, cuBLAS, cuSOLVER, cuSPARSE, and cuRAND from NVIDIA's
CUDA 13.3 PyPI packages, plus the `cutensor-cu13` Windows wheel — the PyPI
toolchain direction established in #1023, without adding CUDA to
`pyproject.toml`'s dependencies. No system CUDA Toolkit is required; a
compatible driver and GPU are needed only to *run* GPU code, not to compile it.

```powershell
pixi install --environment cuda
pixi run --environment cuda check-cuda-layout   # inspect the installed wheels
pixi run --environment cuda cuda-doctor
pixi run --environment cuda build-cuda
pixi run --environment cuda install-python-cuda # editable CUDA-enabled install
```

`build-cuda` configures the `mkl-cuda-windows` preset, which is the CUDA preset
with `USE_CUQUANTUM` off: NVIDIA publishes no Windows build of cuTensorNet or
cuStateVec (#1111). cuTENSOR does ship one and stays enabled. CUDA compilation
is capped at two parallel jobs because `cudafe++` can use substantial memory on
this template-heavy codebase.

Use `pixi shell --environment cuda` for an interactive shell with the CUDA DLL
directories already on `PATH`.
