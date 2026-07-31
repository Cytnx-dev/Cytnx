#!/usr/bin/env python3
"""Run pytest against the extension built by a given CMake preset.

    python tools/run_pytest.py <preset> [pytest args...]

Importing cytnx's extension needs preparation that differs by platform and by
preset, and both kinds have to happen before the interpreter loads it. This
script is what pixi.toml's test-python task goes through so the preparation
stays in one place instead of being spelled out per task.

Windows: since Python 3.8 the extension-module loader searches the system
directories, the directory holding the extension, and directories registered
with ``os.add_dll_directory()`` -- and nothing else. ``PATH`` is not consulted,
so an entry there reaches a plain executable such as ``test_main.exe`` but
never an ``import cytnx``. cytnx's dependency DLLs all sit outside what is
searched by default: ``libarpack.dll`` and its mingw runtime under
``Library\\mingw-w64\\bin``, MKL under ``Library\\bin``, and the CUDA runtime
inside the NVIDIA wheels. tools/activate_windows.bat composes
``CYTNX_WINDOWS_DLL_DIRS`` from the same directories it prepends to ``PATH``;
every entry that exists is registered here.

Linux, ``debug-*`` presets only: USE_DEBUG instruments the extension with
AddressSanitizer, and ASan's ``__cxa_throw`` interceptor resolves the real
``__cxa_throw`` from libstdc++, which python never links. The first C++
exception thrown inside the extension -- ``cytnx_error_msg`` surfacing as
``cytnx.CytnxError``, which the error-path tests hit constantly -- would kill
the process with no traceback under pytest's captured file descriptors.
``LD_PRELOAD`` can only be set before the process starts, so this script
re-executes itself once with both libraries preloaded.

The preloaded libraries come from ``$CONDA_PREFIX/lib``, which is what this
environment actually loads: conda-forge points RPATH at the prefix, so the
prefix carries a newer libstdc++ than the compiler's own private directory,
and preloading the compiler's copy would satisfy the ``libstdc++.so.6`` soname
with a library missing the newer ``GLIBCXX_3.4`` symbol versions.
"""

import os
import runpy
import sys

DLL_DIRS_ENV_VAR = "CYTNX_WINDOWS_DLL_DIRS"
# Set on the re-executed process so the preload is arranged exactly once.
REEXEC_MARKER_ENV_VAR = "CYTNX_PYTEST_PRELOADED"
DEBUG_PRESET_PREFIX = "debug-"
# LeakSanitizer reports CPython's deliberate non-cleanup at shutdown as
# hundreds of leaks, so it is off for an interpreter-hosted run. The C++ suite
# keeps leak detection on, which is where a real cytnx leak surfaces. ASan
# takes the last value of a repeated option, so appending overrides whatever
# the environment already set.
PYTEST_ASAN_OPTIONS = "detect_leaks=0"


def register_dll_directories() -> None:
    """Register each existing directory named by CYTNX_WINDOWS_DLL_DIRS.

    Absent directories are skipped: the variable lists the CUDA locations
    unconditionally, and they exist only once a CUDA layout has been prepared.
    """
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:
        return
    for entry in os.environ.get(DLL_DIRS_ENV_VAR, "").split(os.pathsep):
        if entry and os.path.isdir(entry):
            add_dll_directory(entry)


def reexec_with_asan_preload(argv: list[str]) -> None:
    """Re-run this script with the sanitizer runtime preloaded.

    Returns normally when no preload is needed, and does not return when one
    is: the process is replaced.
    """
    if sys.platform != "linux" or os.environ.get(REEXEC_MARKER_ENV_VAR):
        return
    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        raise SystemExit(
            "run_pytest.py: a debug-* preset needs the sanitizer runtime "
            "preloaded, but CONDA_PREFIX is unset -- run this through "
            "`pixi run test-python` so the environment is activated."
        )
    preload = [
        os.path.join(prefix, "lib", "libasan.so"),
        os.path.join(prefix, "lib", "libstdc++.so"),
    ]
    missing = [path for path in preload if not os.path.exists(path)]
    if missing:
        raise SystemExit(
            "run_pytest.py: cannot preload the sanitizer runtime, missing "
            + ", ".join(missing)
        )

    env = dict(os.environ)
    env[REEXEC_MARKER_ENV_VAR] = "1"
    env["LD_PRELOAD"] = " ".join(preload + [env.get("LD_PRELOAD", "")]).strip()
    asan_options = env.get("ASAN_OPTIONS", "")
    env["ASAN_OPTIONS"] = (
        f"{asan_options}:{PYTEST_ASAN_OPTIONS}" if asan_options else PYTEST_ASAN_OPTIONS
    )
    command = [sys.executable, os.path.abspath(__file__), *argv]
    os.execve(sys.executable, command, env)


def main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit("usage: run_pytest.py <preset> [pytest args...]")
    preset, pytest_args = argv[0], argv[1:]

    if preset.startswith(DEBUG_PRESET_PREFIX):
        if sys.platform == "darwin":
            raise SystemExit(
                "run_pytest.py: the AddressSanitizer preload is a Linux "
                "mechanism; macOS would need DYLD_INSERT_LIBRARIES pointed at "
                "Clang's ASan dylib, which cytnx does not wire up. Run the "
                "Python tests against a release preset instead."
            )
        reexec_with_asan_preload(argv)

    register_dll_directories()
    sys.argv = ["pytest", *pytest_args]
    runpy.run_module("pytest", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main(sys.argv[1:])
