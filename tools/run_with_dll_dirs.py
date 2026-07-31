#!/usr/bin/env python3
"""Run a command with the environment's dependency DLL directories registered.

Windows only. Since Python 3.8 the extension-module loader searches the system
directories, the directory holding the extension itself, and directories
registered with ``os.add_dll_directory()`` -- and nothing else. ``PATH`` is not
consulted, so an entry there reaches a plain executable such as
``test_main.exe`` but never an ``import cytnx``.

cytnx's dependency DLLs all sit outside what is searched by default:
``libarpack.dll`` and its mingw runtime under ``Library\\mingw-w64\\bin``, MKL
under ``Library\\bin``, and, in a CUDA environment, the runtime shipped inside
the NVIDIA wheels' site-packages directories. ``cytnx/__init__.py`` imports the
extension at import time, so without those directories registered the import
fails however ``PATH`` is set.

tools/activate_windows.bat composes ``CYTNX_WINDOWS_DLL_DIRS`` from the same
directories it prepends to ``PATH``, which keeps the list in one place. This
script registers every entry that exists and then runs the command that
follows, in the same process:

    python tools/run_with_dll_dirs.py -m pytest pytests --doctest-modules
"""

import os
import runpy
import sys

DLL_DIRS_ENV_VAR = "CYTNX_WINDOWS_DLL_DIRS"


def register_dll_directories() -> None:
    """Register each existing directory named by the environment variable.

    Absent directories are skipped rather than reported: the variable lists
    the CUDA locations unconditionally, and a CPU environment has none of
    them.
    """
    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:
        return
    for entry in os.environ.get(DLL_DIRS_ENV_VAR, "").split(os.pathsep):
        if entry and os.path.isdir(entry):
            add_dll_directory(entry)


def main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit(
            "usage: run_with_dll_dirs.py [-m MODULE | SCRIPT] [ARGS...]"
        )
    register_dll_directories()
    if argv[0] == "-m":
        if len(argv) < 2:
            raise SystemExit("run_with_dll_dirs.py: -m requires a module name")
        sys.argv = argv[1:]
        runpy.run_module(argv[1], run_name="__main__", alter_sys=True)
    else:
        sys.argv = argv
        runpy.run_path(argv[0], run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1:])
