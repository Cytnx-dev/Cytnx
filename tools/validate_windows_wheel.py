#!/usr/bin/env python3
"""Validate an installed Windows wheel without development DLL roots."""

from __future__ import annotations

import fnmatch
from importlib import metadata
import os
from pathlib import Path
import subprocess
import sys


EXTERNAL_CUDA_DLLS = (
    "cudart64_*.dll",
    "cublas64_*.dll",
    "cublaslt64_*.dll",
    "cusparse64_*.dll",
    "curand64_*.dll",
    "cusolver64_*.dll",
    "cusolvermg64_*.dll",
    "nvrtc64_*.dll",
    "nvrtc-builtins64_*.dll",
    "nvjitlink_*.dll",
    "nvvm64_*.dll",
    "cutensor*.dll",
    "nvcuda.dll",
)

CUDA_DISTRIBUTIONS = (
    "nvidia-cuda-runtime",
    "nvidia-cublas",
    "nvidia-cusparse",
    "nvidia-curand",
    "nvidia-cusolver",
    "cutensor-cu13",
)


def _installed_distribution() -> tuple[str, metadata.Distribution]:
    installed = []
    for name in ("cytnx-cuda", "cytnx"):
        try:
            installed.append((name, metadata.distribution(name)))
        except metadata.PackageNotFoundError:
            pass
    if len(installed) != 1:
        raise RuntimeError(f"expected exactly one Cytnx distribution, found {installed}")
    return installed[0]


def _clear_development_roots() -> None:
    prefixes = []
    for key in (
        "CONDA_PREFIX",
        "CUDA_PATH",
        "CUDA_HOME",
        "CUDAToolkit_ROOT",
        "CUTENSOR_ROOT",
        "CUQUANTUM_ROOT",
    ):
        value = os.environ.pop(key, None)
        if value:
            prefixes.append(os.path.normcase(os.path.abspath(value)))

    clean_path = []
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        normalized = os.path.normcase(os.path.abspath(entry)) if entry else ""
        if any(normalized == root or normalized.startswith(root + os.sep) for root in prefixes):
            continue
        if "nvidia\\cu13" in normalized or "cutensor" in normalized:
            continue
        clean_path.append(entry)
    os.environ["PATH"] = os.pathsep.join(clean_path)


def main() -> None:
    if sys.platform != "win32":
        raise SystemExit("this validation is only meaningful on Windows")
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    distribution_name, distribution = _installed_distribution()
    is_cuda = distribution_name == "cytnx-cuda"

    if is_cuda:
        for dependency in CUDA_DISTRIBUTIONS:
            metadata.distribution(dependency)
        bundled_names = [Path(str(path)).name.lower() for path in distribution.files or ()]
        unexpectedly_bundled = sorted(
            name
            for name in bundled_names
            if any(fnmatch.fnmatch(name, pattern) for pattern in EXTERNAL_CUDA_DLLS)
        )
        if unexpectedly_bundled:
            raise RuntimeError(
                "CUDA/cuTENSOR DLLs must remain in their declared PyPI packages: "
                f"{unexpectedly_bundled}"
            )

    _clear_development_roots()
    import cytnx

    tensor = cytnx.ones([2, 3])
    if list(tensor.shape()) != [2, 3]:
        raise RuntimeError(f"unexpected tensor shape: {tensor.shape()}")
    if os.environ.get("CYTNX_EXPECT_NO_GPU") == "1" and cytnx.Device.Ngpus != 0:
        raise RuntimeError(f"expected a GPU-less runner, found {cytnx.Device.Ngpus} GPUs")
    print(
        f"validated {distribution_name} {distribution.version} with Python "
        f"{sys.version.split()[0]} (Ngpus={cytnx.Device.Ngpus})"
    )


if __name__ == "__main__":
    main()
