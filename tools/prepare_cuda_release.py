#!/usr/bin/env python3
"""Rewrite pyproject.toml in place for a cytnx-cuda release build.

The script

  1. renames the PyPI project from `cytnx` to `cytnx-cuda`;
  2. adds the NVIDIA/cuTENSOR/cuQuantum runtime packages as ordinary
     `[project.dependencies]`, pinned with PEP 440 compatible-release
     (`~=`) constraints anchored to the versions this wheel was built
     and linked against (the PyTorch approach: the CUDA libraries are
     dynamically linked at runtime from these pip packages, never
     vendored into the wheel itself), allowing patch-level upgrades
     within the same minor version while blocking anything that could
     break binary compatibility. Only the packages
     cytnx links directly are listed; each pulls in its own transitive
     NVIDIA dependencies (nvidia-cusolver -> nvidia-cublas/
     nvidia-nvjitlink/nvidia-cusparse, nvidia-cublas -> nvidia-cuda-nvrtc,
     cutensornet-cu13 -> cutensor-cu13) via pip's own resolver, so they
     don't need to be listed here too. cuquantum-cu13 itself is
     deliberately NOT depended on: it pulls in cudensitymat/cupauliprop/
     custabilizer, none of which cytnx links;
  3. switches `[tool.scikit-build].cmake.args` to the `openblas-cuda`
     preset; and
  4. overrides `[tool.cibuildwheel.linux].repair-wheel-command` to
     exclude those same libraries from `auditwheel repair`'s vendoring,
     so the wheel stays small and relies on the pip dependencies at
     import time (see cytnx/_cuda_preload.py for how they get resolved
     without requiring LD_LIBRARY_PATH).

This is intended to run inside a fresh CI checkout before cibuildwheel,
after (or before -- the two don't conflict) tools/prepare_nightly_release.py
for nightly builds. It mutates the working tree and is not idempotent.

Requires `tomlkit` (declared in pyproject.toml's `release-tools`
dependency-group) so the rewrite preserves comments and formatting on
round-trip.
"""

import pathlib

import tomlkit

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"

# PEP 440 compatible-release ("~=") constraints anchored to the versions
# this wheel is built and linked against -- patch releases within the
# same minor version are allowed to satisfy the dependency, anything
# outside it is not (see the module docstring for why only these, not
# their transitive NVIDIA/cuTENSOR dependencies, need listing).
CUDA_RUNTIME_DEPENDENCIES = [
    "nvidia-cuda-runtime ~=13.3.29 ; sys_platform == 'linux'",
    "nvidia-cublas ~=13.6.0.2 ; sys_platform == 'linux'",
    "nvidia-cusparse ~=12.8.2.51 ; sys_platform == 'linux'",
    "nvidia-curand ~=10.4.3.29 ; sys_platform == 'linux'",
    "nvidia-cusolver ~=12.2.6.9 ; sys_platform == 'linux'",
    "cutensor-cu13 ~=2.7.0 ; sys_platform == 'linux'",
    "cutensornet-cu13 ~=2.13.0 ; sys_platform == 'linux'",
    "custatevec-cu13 ~=1.14.0 ; sys_platform == 'linux'",
]

# SONAMEs provided by the pip packages above (including their transitive
# NVIDIA dependencies -- nvidia-cuda-nvrtc via nvidia-cublas,
# nvidia-nvjitlink via nvidia-cusolver), excluded from auditwheel's
# vendoring so the wheel relies on them at runtime instead of bundling
# its own copies.
EXCLUDED_SONAMES = [
    "libcudart.so.*",
    "libcublas.so.*",
    "libcublasLt.so.*",
    "libnvrtc.so.*",
    "libcusparse.so.*",
    "libcurand.so.*",
    "libcusolver.so.*",
    "libnvJitLink.so.*",
    "libcutensor.so.*",
    "libcutensorMg.so.*",
    "libcutensornet.so.*",
    "libcustatevec.so.*",
]


def rewrite_pyproject(doc: tomlkit.TOMLDocument) -> None:
    project = doc["project"]
    project["name"] = "cytnx-cuda"

    dependencies = list(project.get("dependencies", []))
    dependencies.extend(CUDA_RUNTIME_DEPENDENCIES)
    project["dependencies"] = dependencies

    skb = doc["tool"]["scikit-build"]
    if "--preset=openblas-cpu" not in list(skb["cmake"]["args"]):
        raise SystemExit(
            "expected [tool.scikit-build].cmake.args to still be "
            "--preset=openblas-cpu before this rewrite; pyproject.toml's "
            "structure may have changed"
        )
    skb["cmake"]["args"] = ["--preset=openblas-cuda"]

    exclude_flags = " ".join(f"--exclude {name}" for name in EXCLUDED_SONAMES)
    linux = doc["tool"]["cibuildwheel"]["linux"]
    linux["repair-wheel-command"] = (
        f"auditwheel repair {exclude_flags} -w {{dest_dir}} {{wheel}}"
    )

    PYPROJECT.write_text(tomlkit.dumps(doc))


def main() -> None:
    doc = tomlkit.parse(PYPROJECT.read_text())
    rewrite_pyproject(doc)
    print("stamped pyproject.toml for cytnx-cuda")


if __name__ == "__main__":
    main()
