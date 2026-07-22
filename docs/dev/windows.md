# Developing Cytnx on Windows

Native Windows development uses Pixi for Python 3.10, CMake, Ninja, MKL, and
the other build dependencies. CMake drives the Microsoft compiler (`cl.exe`)
through the Ninja generator.

## Prerequisites

Install Visual Studio 2022 or Visual Studio 2022 Build Tools with:

- Desktop development with C++
- MSVC v143 x64/x86 build tools
- A Windows 10 or Windows 11 SDK

Pixi's `c-compiler` and `cxx-compiler` packages locate and activate that
installation; they do not contain MSVC itself.

Install Pixi and make `pixi.exe` available on `PATH`. You can also invoke it
by its full path if the current terminal has not picked up a recent PATH
change.

## CPU build

From an x64 PowerShell prompt at the repository root:

```powershell
pixi install
pixi run doctor
pixi run build
```

The default environment is locked to Python 3.10 and builds with MSVC, Ninja,
and MKL. The first build initializes the git submodules automatically.

Run the C++ tests with:

```powershell
pixi run test-cpp
```

To build and install the Python extension in editable mode, then run its
tests:

```powershell
pixi run install-python
pixi run test-python
```

## CUDA build from PyPI packages

CUDA is an optional Pixi feature rather than a base project dependency. It
uses NVIDIA's CUDA 13.3 PyPI packages for nvcc, CCCL, the runtime, cuBLAS,
cuSOLVER, cuSPARSE, and cuRAND, plus the Windows `cutensor-cu13` wheel. This
follows the PyPI toolchain layout introduced in PR #1023 without changing the
normal dependencies in `pyproject.toml`.

```powershell
pixi install --environment cuda
pixi run --environment cuda cuda-doctor
pixi run --environment cuda build-cuda
```

The compiler and libraries live under the CUDA environment's
`Lib\site-packages` directory; no system CUDA Toolkit is required. A compatible
NVIDIA driver and GPU are needed to execute GPU code, but not to configure and
compile the CUDA build. CUDA compilation is limited to two parallel jobs by
default because `cudafe++` can use substantial memory on this template-heavy
codebase.

NVIDIA's Windows math-library wheels currently omit the small MSVC import
libraries that accompany their DLLs. The `prepare-cuda` dependency of the
configure task derives those local `.lib` files from the installed DLL export
tables with MSVC's `lib.exe`; it does not download or vendor another CUDA
toolkit. Use the following command to inspect the installed wheel layout
without compiling:

```powershell
pixi run --environment cuda check-cuda-layout
```

cuQuantum is disabled in the native Windows CUDA task because its cuTensorNet
and cuStateVec packages do not currently publish Windows wheels. cuTENSOR is
enabled because `cutensor-cu13` does publish a Windows wheel.

To produce an editable CUDA-enabled Python install:

```powershell
pixi run --environment cuda install-python-cuda
```

Use `pixi shell --environment cuda` when you want an interactive shell with
the CUDA DLL directories already on `PATH`.
