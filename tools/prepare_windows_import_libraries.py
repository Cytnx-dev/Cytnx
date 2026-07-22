#!/usr/bin/env python3
"""Create MSVC import libraries omitted by Windows dependency packages.

The conda-forge ARPACK package provides a MinGW DLL and GNU import archive.
NVIDIA's CUDA 13 PyPI runtime wheels provide DLLs and headers, but several
math-library wheels omit their MSVC import libraries. CMake needs MSVC import
libraries in both cases.

This script reads each installed DLL's PE export table and invokes MSVC's
``lib.exe`` to create a local import library inside the disposable Pixi
environment. It does not copy or vendor any dependency and is idempotent.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import pefile


CUDA_IMPORT_LIBRARIES = {
    "cublas.lib": "cublas64_*.dll",
    "cublasLt.lib": "cublasLt64_*.dll",
    "curand.lib": "curand64_*.dll",
    "cusolver.lib": "cusolver64_*.dll",
    "cusolverMg.lib": "cusolverMg64_*.dll",
    "cusparse.lib": "cusparse64_*.dll",
}


def _find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {pattern!r} below {directory}, found {matches}"
        )
    return matches[0]


def _exports(dll: Path) -> list[str]:
    image = pefile.PE(str(dll), fast_load=True)
    try:
        image.parse_data_directories(
            directories=[
                pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_EXPORT"]
            ]
        )
        export_table = getattr(image, "DIRECTORY_ENTRY_EXPORT", None)
        if export_table is None:
            raise RuntimeError(f"{dll} has no PE export table")
        names = {
            symbol.name.decode("ascii")
            for symbol in export_table.symbols
            if symbol.name is not None
        }
        # MinGW DLLs can expose linker bookkeeping entries alongside their
        # public API. Feeding those reserved archive member names back to
        # lib.exe creates an archive that MSVC subsequently rejects as corrupt.
        names = sorted(
            name
            for name in names
            if name != "__NULL_IMPORT_DESCRIPTOR"
            and not name.startswith("__IMPORT_DESCRIPTOR_")
        )
    finally:
        image.close()
    if not names:
        raise RuntimeError(f"{dll} has no named exports")
    return names


def _write_definition(path: Path, dll: Path, exports: list[str]) -> None:
    contents = [f'LIBRARY "{dll.name}"', "EXPORTS"]
    contents.extend(f"  {name}" for name in exports)
    path.write_text("\n".join(contents) + "\n", encoding="ascii")


def _prepare_one(
    dll: Path,
    output: Path,
    *,
    check_only: bool,
    librarian: str | None,
    temporary_directory: Path,
) -> None:
    if not dll.is_file():
        raise RuntimeError(f"dependency DLL not found: {dll}")
    exports = _exports(dll)
    if check_only:
        state = "present" if output.is_file() else "will generate"
        print(f"{output.name}: {len(exports)} exports from {dll.name} ({state})")
        return
    if output.is_file():
        validation = subprocess.run(
            [librarian, "/nologo", "/list", str(output)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if validation.returncode == 0:
            output.with_suffix(".exp").unlink(missing_ok=True)
            print(f"{output.name}: already present")
            return
        print(f"{output.name}: replacing invalid import library")
        output.unlink()
    if librarian is None:
        raise AssertionError("librarian is required outside check mode")

    output.parent.mkdir(parents=True, exist_ok=True)
    definition = temporary_directory / f"{output.stem}.def"
    _write_definition(definition, dll, exports)
    subprocess.run(
        [
            librarian,
            "/nologo",
            f"/def:{definition}",
            f"/out:{output}",
            "/machine:x64",
        ],
        check=True,
    )
    output.with_suffix(".exp").unlink(missing_ok=True)
    print(f"{output.name}: generated from {dll.name}")


def prepare(*, include_cuda: bool, check_only: bool) -> None:
    conda_prefix_value = os.environ.get("CONDA_PREFIX")
    if not conda_prefix_value:
        raise RuntimeError("CONDA_PREFIX is unset; run this script through Pixi")
    conda_prefix = Path(conda_prefix_value).resolve()

    librarian = None if check_only else shutil.which("lib.exe")
    if not check_only and librarian is None:
        raise RuntimeError(
            "lib.exe is not on PATH. Install Visual Studio 2022 Build Tools "
            "with the Desktop development with C++ workload, then run this "
            "command through Pixi."
        )

    with tempfile.TemporaryDirectory(prefix="cytnx-import-") as temporary:
        temporary_directory = Path(temporary)
        _prepare_one(
            conda_prefix / "Library" / "mingw-w64" / "bin" / "libarpack.dll",
            conda_prefix / "Library" / "lib" / "arpack.lib",
            check_only=check_only,
            librarian=librarian,
            temporary_directory=temporary_directory,
        )

        if not include_cuda:
            return
        cuda_path_value = os.environ.get("CUDA_PATH")
        if not cuda_path_value:
            raise RuntimeError(
                "CUDA_PATH is unset; select the Pixi cuda environment"
            )
        cuda_root = Path(cuda_path_value).resolve()
        dll_directory = cuda_root / "bin" / "x86_64"
        library_directory = cuda_root / "lib" / "x64"
        if not (cuda_root / "bin" / "nvcc.exe").is_file():
            raise RuntimeError(f"nvcc.exe was not found below CUDA root {cuda_root}")
        if not dll_directory.is_dir() or not library_directory.is_dir():
            raise RuntimeError(f"incomplete CUDA PyPI layout below {cuda_root}")

        for import_name, dll_pattern in CUDA_IMPORT_LIBRARIES.items():
            _prepare_one(
                _find_one(dll_directory, dll_pattern),
                library_directory / import_name,
                check_only=check_only,
                librarian=librarian,
                temporary_directory=temporary_directory,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="also prepare CUDA math-library imports from the PyPI wheels",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate dependency layouts without requiring lib.exe",
    )
    args = parser.parse_args()

    try:
        prepare(include_cuda=args.cuda, check_only=args.check)
    except (OSError, RuntimeError, pefile.PEFormatError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
