#!/usr/bin/env python3
"""Repair a Windows wheel while keeping CUDA runtime DLLs in PyPI packages."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
import os
from pathlib import Path
import subprocess
import sys
import zipfile


CUDA_DLL_PATTERNS = (
    "cudart64_*.dll",
    "cublas64_*.dll",
    "cublasLt64_*.dll",
    "cusparse64_*.dll",
    "curand64_*.dll",
    "cusolver64_*.dll",
    "cusolverMg64_*.dll",
    "nvrtc64_*.dll",
    "nvrtc-builtins64_*.dll",
    "nvJitLink_*.dll",
    "nvvm64_*.dll",
    "cutensor*.dll",
    # This is supplied by the NVIDIA display driver, never by a Python wheel.
    "nvcuda.dll",
)


def _wheel_distribution_name(wheel: Path) -> str:
    with zipfile.ZipFile(wheel) as archive:
        metadata_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise RuntimeError(
                f"expected one dist-info/METADATA in {wheel}, found {metadata_files}"
            )
        metadata = BytesParser().parsebytes(archive.read(metadata_files[0]))
    name = metadata.get("Name")
    if not name:
        raise RuntimeError(f"wheel metadata in {wheel} has no Name field")
    return name.lower().replace("_", "-")


def repair(wheel: Path, destination: Path) -> None:
    if sys.platform != "win32":
        raise RuntimeError("Windows wheel repair must run on Windows")
    prefix_value = os.environ.get("CONDA_PREFIX")
    if not prefix_value:
        raise RuntimeError("CONDA_PREFIX is unset; activate the Pixi wheel environment")

    prefix = Path(prefix_value).resolve()
    search_paths = [
        prefix / "Library" / "bin",
        prefix / "Library" / "mingw-w64" / "bin",
    ]
    is_cuda = _wheel_distribution_name(wheel) == "cytnx-cuda"
    if is_cuda:
        site_packages = prefix / "Lib" / "site-packages"
        search_paths.extend(
            [
                site_packages / "nvidia" / "cu13" / "bin" / "x86_64",
                site_packages / "cutensor" / "bin",
            ]
        )

    missing = [str(path) for path in search_paths if not path.is_dir()]
    if missing:
        raise RuntimeError(f"wheel repair search directories are missing: {missing}")

    destination.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "delvewheel",
        "repair",
        "--wheel-dir",
        str(destination),
        "--add-path",
        ";".join(str(path) for path in search_paths),
        # conda-forge's libblas.dll and liblapack.dll export symbols through
        # literal openblas.dll forwarders. delvewheel rewrites import tables,
        # but not PE export-forwarder strings, so this basename must remain
        # stable in the repaired wheel.
        "--no-mangle",
        "openblas.dll",
    ]
    if is_cuda:
        command.extend(["--exclude", ";".join(CUDA_DLL_PATTERNS)])
    command.append(str(wheel))
    print("Running:", subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--dest-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        repair(args.wheel.resolve(), args.dest_dir.resolve())
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
