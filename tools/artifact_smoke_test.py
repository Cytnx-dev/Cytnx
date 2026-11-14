#!/usr/bin/env python3
"""Smoke-test an installed Cytnx package: a wheel or a conda archive.

The release pipelines run this against the artifact they are about to
publish, installed into a clean environment, so that an archive which cannot
be imported, carries the wrong version, or computes wrong numbers fails the
build instead of reaching users::

    python tools/artifact_smoke_test.py --version-cmake version.cmake

It is a plain script rather than a pytest module on purpose. ``import cytnx``
is the first non-standard-library import of a fresh interpreter, so nothing a
test runner (or numpy) loaded earlier can satisfy a shared-library dependency
the package itself fails to declare or bundle.

Checks, in order:

  1. ``import cytnx`` succeeds, and the module was not picked up from the
     source checkout (which would test the tree, not the artifact).
  2. The version agrees everywhere: ``cytnx.__version__`` (compiled into the
     extension), the installed distribution metadata, and, with
     ``--check-conda-metadata``, the conda package record, all equal the
     expected version. The expected version is ``--expected-version`` or, by
     default, ``MAJOR.MINOR.PATCH`` parsed from ``--version-cmake`` plus the
     ``CYTNX_VERSION_TAG`` suffix the nightly builds set.
  3. A handful of numerical operations give results worked out by hand (see
     each check's comment). They are chosen to reach the native libraries an
     artifact has to ship or link correctly: BLAS (``Matmul``), LAPACK
     (``Eigh``, ``Svd``), HPTT (``permute``) and the numpy bridge. None of the
     expected values comes from another Cytnx code path.

With ``--device cuda`` the numerical checks run on the first GPU, which is how
a CUDA wheel is exercised on a GPU runner.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import pathlib
import re
import sys

VERSION_TAG_ENV_VAR = "CYTNX_VERSION_TAG"
# The same sources are published under two distribution names: tools/
# prepare_cuda_release.py renames the CUDA build. Exactly one may be installed.
DISTRIBUTION_NAMES = ("cytnx", "cytnx-cuda")
# Every check runs in double precision on numbers of order 1 to 50.
ABS_TOL = 1e-12


def _fail(message: str) -> None:
    raise SystemExit(f"artifact_smoke_test: FAILED: {message}")


def read_cmake_version(version_cmake: pathlib.Path) -> str:
    """Return ``MAJOR.MINOR.PATCH`` from a version.cmake file.

    The pattern mirrors the one scikit-build-core uses in pyproject.toml's
    ``[tool.scikit-build.metadata.version]`` block, so this accepts exactly
    the version.cmake the package build accepts.

    Args:
        version_cmake: Path of the version.cmake file to parse.

    Returns:
        The numeric version string, for example ``"1.1.1"``.
    """
    text = version_cmake.read_text()
    parts = []
    for field in ("MAJOR", "MINOR", "PATCH"):
        match = re.search(rf"set\(\w+?VERSION_{field}\s+(\d+)\)", text)
        if not match:
            _fail(f"could not parse {field} version from {version_cmake}")
        parts.append(match.group(1))
    return ".".join(parts)


def check_not_from_checkout(cytnx, checkout: pathlib.Path | None) -> None:
    """Fail if ``cytnx`` was imported from the source tree at ``checkout``."""
    module_dir = pathlib.Path(cytnx.__file__).resolve().parent
    print(f"cytnx imported from {module_dir}")
    if checkout is not None and module_dir == checkout.resolve() / "cytnx":
        _fail(
            f"cytnx was imported from the source checkout ({module_dir}), "
            f"not from the installed artifact"
        )


def installed_distribution() -> importlib.metadata.Distribution:
    """Return the one installed distribution that provides ``cytnx``."""
    installed = []
    for name in DISTRIBUTION_NAMES:
        try:
            installed.append(importlib.metadata.distribution(name))
        except importlib.metadata.PackageNotFoundError:
            pass
    if len(installed) != 1:
        names = [distribution.metadata["Name"] for distribution in installed]
        _fail(
            f"expected exactly one of {DISTRIBUTION_NAMES} to be installed, "
            f"found {names}"
        )
    return installed[0]


def read_conda_versions(distribution: str) -> list[str]:
    """Return the versions conda recorded for a package in this prefix."""
    conda_meta = pathlib.Path(sys.prefix) / "conda-meta"
    versions = []
    for record_path in sorted(conda_meta.glob(f"{distribution}-*.json")):
        record = json.loads(record_path.read_text())
        if record.get("name") == distribution:
            versions.append(str(record.get("version")))
    return versions


def check_versions(cytnx, expected: str, check_conda_metadata: bool) -> str:
    """Fail unless every version the artifact reports equals ``expected``.

    Returns:
        The name of the installed distribution that was checked.
    """
    distribution = installed_distribution()
    name = distribution.metadata["Name"]
    reported = {
        "cytnx.__version__": cytnx.__version__,
        f"distribution {name!r}": distribution.version,
    }
    if check_conda_metadata:
        conda_versions = read_conda_versions(name)
        if len(conda_versions) != 1:
            _fail(
                f"expected exactly one conda record for {name!r} in "
                f"{sys.prefix}/conda-meta, found {conda_versions}"
            )
        reported["conda package record"] = conda_versions[0]
    for source, version in reported.items():
        print(f"{source} = {version}")
    wrong = {
        source: version
        for source, version in reported.items()
        if version != expected
    }
    if wrong:
        _fail(f"expected version {expected!r} everywhere, but got {wrong}")
    return name


def _assert_close(name: str, actual, expected) -> None:
    """Compare two equally nested lists of real or complex numbers."""
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            _fail(f"{name}: expected {expected}, got {actual}")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected)
        ):
            _assert_close(f"{name}[{index}]", actual_item, expected_item)
        return
    if not math.isfinite(abs(actual)) or abs(actual - expected) > ABS_TOL:
        _fail(f"{name}: expected {expected}, got {actual}")


def check_numerics(cytnx, device: int) -> None:
    """Run small computations with hand-derived results on ``device``."""
    # Imported here, after cytnx: see the module docstring.
    import numpy as np

    def on_device(values, dtype):
        return cytnx.from_numpy(np.array(values, dtype=dtype)).to(device)

    def to_list(tensor):
        return tensor.to(cytnx.Device.cpu).numpy().tolist()

    # BLAS gemm. [[1,2],[3,4]] @ [[5,6],[7,8]]:
    #   row 0: 1*5+2*7 = 19, 1*6+2*8 = 22
    #   row 1: 3*5+4*7 = 43, 3*6+4*8 = 50
    lhs = on_device([[1.0, 2.0], [3.0, 4.0]], np.float64)
    rhs = on_device([[5.0, 6.0], [7.0, 8.0]], np.float64)
    product = cytnx.linalg.Matmul(lhs, rhs)
    if product.dtype() != cytnx.Type.Double:
        _fail(f"Matmul(Double, Double) returned dtype {product.dtype_str()}")
    _assert_close("Matmul", to_list(product), [[19.0, 22.0], [43.0, 50.0]])

    # LAPACK symmetric eigensolver. [[2,1],[1,2]] has characteristic
    # polynomial (2-x)^2 - 1, so its eigenvalues are 1 and 3 (ascending).
    symmetric = on_device([[2.0, 1.0], [1.0, 2.0]], np.float64)
    eigenvalues = cytnx.linalg.Eigh(symmetric)[0]
    _assert_close("Eigh(real)", to_list(eigenvalues), [1.0, 3.0])

    # Complex Hermitian eigensolver and complex gemm on the Pauli-Y matrix
    # [[0,-i],[i,0]]: eigenvalues -1 and +1, and Y @ Y is the identity.
    pauli_y = on_device([[0.0, -1.0j], [1.0j, 0.0]], np.complex128)
    eigenvalues = cytnx.linalg.Eigh(pauli_y)[0]
    _assert_close("Eigh(complex)", to_list(eigenvalues), [-1.0, 1.0])
    _assert_close(
        "Matmul(complex)",
        to_list(cytnx.linalg.Matmul(pauli_y, pauli_y)),
        [[1.0 + 0.0j, 0.0j], [0.0j, 1.0 + 0.0j]],
    )

    # LAPACK SVD. For A = [[3,0],[4,5]], A^T A = [[25,20],[20,25]] has
    # eigenvalues 45 and 5, so the singular values are sqrt(45) and sqrt(5).
    triangular = on_device([[3.0, 0.0], [4.0, 5.0]], np.float64)
    singular_values = cytnx.linalg.Svd(triangular)[0]
    _assert_close(
        "Svd", to_list(singular_values), [math.sqrt(45.0), math.sqrt(5.0)]
    )

    # HPTT transpose. T[i,j,k] = 12*i + 4*j + k on shape (2,3,4); after
    # permute(2,0,1) the element at [k,i,j] must be that same number.
    counting = cytnx.arange(24).reshape(2, 3, 4).to(device)
    permuted = counting.permute(2, 0, 1).contiguous()
    if list(permuted.shape()) != [4, 2, 3]:
        _fail(f"permute: expected shape [4, 2, 3], got {permuted.shape()}")
    _assert_close(
        "permute",
        to_list(permuted),
        [
            [[float(12 * i + 4 * j + k) for j in range(3)] for i in range(2)]
            for k in range(4)
        ],
    )

    # UniTensor contraction over the shared label "b": the same matrix
    # product as above, reached through the tensor-network layer.
    left = cytnx.UniTensor(lhs, rowrank=1).relabel(["a", "b"])
    right = cytnx.UniTensor(rhs, rowrank=1).relabel(["b", "c"])
    contracted = cytnx.Contract(left, right)
    if list(contracted.labels()) != ["a", "c"]:
        _fail(f"Contract: expected labels a, c; got {contracted.labels()}")
    _assert_close(
        "Contract",
        to_list(contracted.get_block()),
        [[19.0, 22.0], [43.0, 50.0]],
    )
    print("numerical smoke test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test an installed Cytnx wheel or conda package."
    )
    version_source = parser.add_mutually_exclusive_group(required=True)
    version_source.add_argument(
        "--expected-version",
        help="exact version every part of the artifact must report",
    )
    version_source.add_argument(
        "--version-cmake",
        type=pathlib.Path,
        help="version.cmake to take the expected version from; the "
        f"{VERSION_TAG_ENV_VAR} environment variable, when set, is appended. "
        "Its directory is treated as the source checkout cytnx must not be "
        "imported from.",
    )
    parser.add_argument(
        "--check-conda-metadata",
        action="store_true",
        help="also require the conda package record of this environment to "
        "carry the expected version",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="where the numerical checks run; cuda needs a visible GPU",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkout = None
    expected = args.expected_version
    if args.version_cmake is not None:
        checkout = args.version_cmake.resolve().parent
        expected = read_cmake_version(args.version_cmake) + os.environ.get(
            VERSION_TAG_ENV_VAR, ""
        )

    # Deliberately the first import outside the standard library.
    import cytnx

    check_not_from_checkout(cytnx, checkout)
    distribution = check_versions(cytnx, expected, args.check_conda_metadata)

    device = cytnx.Device.cpu
    if args.device == "cuda":
        if cytnx.Device.Ngpus < 1:
            _fail(
                "--device cuda needs a GPU, but Device.Ngpus is "
                f"{cytnx.Device.Ngpus}"
            )
        device = cytnx.Device.cuda
    check_numerics(cytnx, device)
    print(f"artifact_smoke_test: OK ({distribution} {expected}, {args.device})")


if __name__ == "__main__":
    main()
