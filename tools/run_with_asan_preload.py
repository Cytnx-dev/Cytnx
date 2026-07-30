#!/usr/bin/env python3
"""Run a command against an ASan-instrumented cytnx build.

Usage: run_with_asan_preload.py <build-dir> <command> [args...]

A `USE_DEBUG=ON` build instruments cytnx with AddressSanitizer, and the
resulting extension module cannot simply be imported by an ordinary
interpreter. Two things go wrong, both of which this script fixes for the
command it runs, and only for that command:

1. ASan's ``__cxa_throw`` interceptor resolves the real ``__cxa_throw`` from
   libstdc++, but a plain ``python`` never links libstdc++. The first C++
   exception thrown inside the extension -- ``cytnx_error_msg`` surfaces as
   ``cytnx.CytnxError``, which ordinary error-path tests hit constantly --
   then CHECK-fails the interceptor and kills the process instantly, with no
   traceback under pytest's captured file descriptors. Preloading
   ``libasan.so`` together with ``libstdc++.so`` fixes it.
2. LeakSanitizer's default reports hundreds of "leaks" that are CPython's own
   deliberate non-cleanup at interpreter shutdown -- interned strings, static
   type objects, allocator arenas -- none attributable to cytnx.
   ``detect_leaks=0`` turns that off for this interpreter-hosted run alone.
   A ctest run is deliberately left untouched and keeps leak detection on,
   since that is where a real cytnx leak would be caught.

The compiler that actually built the extension is read out of the build's own
CMakeCache.txt rather than guessed: preloading one GCC's libasan.so into a
binary built by a different GCC, or by Clang, is an ASan runtime/ABI
mismatch. ``-print-file-name`` silently echoes its argument back when it
cannot resolve a library, so both paths must come back absolute before
anything is preloaded.

Everything here is Linux/GCC-specific. On any other platform, and for a build
that is not ASan-instrumented, the command runs with no changes at all --
macOS ASan discovery is dylib-based and is not handled.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys


def read_cache_entry(cache: Path, name: str) -> str | None:
    """Return the value of a CMake cache entry, or None if it is absent.

    The entry's type is normally FILEPATH or BOOL, but a build configured with
    an explicit -D records it as STRING instead, so the type is not matched.
    """
    pattern = re.compile(rf"^{re.escape(name)}:[^=]*=(.*)$")
    try:
        for line in cache.read_text(errors="ignore").splitlines():
            match = pattern.match(line)
            if match:
                return match.group(1).strip()
    except OSError:
        return None
    return None


def is_gnu_compiler(compiler: str) -> bool:
    """Whether `compiler` identifies itself as GCC. Clang's --version does not
    carry the Free Software Foundation line."""
    try:
        version = subprocess.run(
            [compiler, "--version"], capture_output=True, text=True, check=False
        )
    except OSError:
        return False
    return "Free Software Foundation" in version.stdout


def library_path(compiler: str, library: str) -> str | None:
    """Resolve `library` through the compiler, or None if it stays relative."""
    try:
        located = subprocess.run(
            [compiler, f"-print-file-name={library}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    path = located.stdout.strip()
    return path if os.path.isabs(path) else None


def asan_environment(build_dir: Path) -> dict[str, str]:
    """The LD_PRELOAD/ASAN_OPTIONS pair this build needs, empty if none."""
    if sys.platform != "linux":
        return {}
    cache = build_dir / "CMakeCache.txt"
    if read_cache_entry(cache, "USE_DEBUG") != "ON":
        return {}
    compiler = read_cache_entry(cache, "CMAKE_CXX_COMPILER")
    if not compiler or not is_gnu_compiler(compiler):
        return {}
    libasan = library_path(compiler, "libasan.so")
    libstdcxx = library_path(compiler, "libstdc++.so")
    if not libasan or not libstdcxx:
        return {}
    return {"LD_PRELOAD": f"{libasan} {libstdcxx}", "ASAN_OPTIONS": "detect_leaks=0"}


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(f"usage: {sys.argv[0]} <build-dir> <command> [args...]")
    build_dir = Path(sys.argv[1])
    command = sys.argv[2:]

    environment = dict(os.environ)
    added = asan_environment(build_dir)
    if added:
        print(f"{sys.argv[0]}: preloading {added['LD_PRELOAD']}", file=sys.stderr)
        environment.update(added)

    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()
