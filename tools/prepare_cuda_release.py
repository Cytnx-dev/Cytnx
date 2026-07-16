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
     preset and sets `CMAKE_INSTALL_RPATH` to an $ORIGIN-relative RUNPATH
     (see CUDA_INSTALL_RPATH) so the installed extension finds those
     libraries in their sibling pip packages at runtime without
     LD_LIBRARY_PATH;
  4. overrides `[tool.cibuildwheel.linux].repair-wheel-command` to
     exclude those same libraries from `auditwheel repair`'s vendoring,
     so the wheel stays small and relies on the pip dependencies (found
     via the RUNPATH from step 3) instead; and
  5. chains tools/cibuildwheel_before_all_cuda.sh onto
     `[tool.cibuildwheel.linux].before-all` and points CMAKE_CUDA_COMPILER/
     CUTENSOR_ROOT/CUQUANTUM_ROOT/PATH/CMAKE_PREFIX_PATH at the toolchain it
     installs, so the build step can compile CUDA/cuTENSOR/cuQuantum code
     without a system-wide CUDA install (see that script's own docstring).

This is intended to run inside a fresh CI checkout before cibuildwheel,
after (or before -- the two don't conflict) tools/prepare_nightly_release.py
for nightly builds. It mutates the working tree and is not idempotent.

Requires `tomlkit` (declared in pyproject.toml's `release-tools`
dependency-group) so the rewrite preserves comments and formatting on
round-trip.
"""

import pathlib
import shlex

import tomlkit

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"

# Where tools/cibuildwheel_before_all_cuda.sh installs the CUDA toolchain
# inside the manylinux container (see that script's docstring for why an
# isolated --target dir rather than the build venv's site-packages).
CUDA_TOOLCHAIN_PREFIX = "/opt/cuda-toolchain"
# All nvidia-* packages share the "nvidia" namespace package, so installing
# them together into CUDA_TOOLCHAIN_PREFIX merges them into one
# nvidia/cu13/{bin,include,lib} tree.
CUDA_TOOLCHAIN_ROOT = f"{CUDA_TOOLCHAIN_PREFIX}/nvidia/cu13"

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

# The compiler toolchain tools/cibuildwheel_before_all_cuda.sh installs to
# CUDA_TOOLCHAIN_PREFIX for the build step. These are build-time-only tools,
# not wheel runtime dependencies, so they aren't in CUDA_RUNTIME_DEPENDENCIES
# above: nvidia-cuda-nvcc is the compiler; nvidia-cuda-cccl provides the CUDA
# C++ Core Libraries headers (thrust/cub/libcudacxx) under
# nvidia/cu13/include/cccl, which CMake's FindCUDAToolkit adds to the
# CUDA::toolkit include interface and errors out on if the directory is
# absent. The environment marker on each CUDA_RUNTIME_DEPENDENCIES entry is
# dropped here since this only ever runs inside the Linux manylinux container.
CUDA_BUILD_TOOLCHAIN = [
    "nvidia-cuda-nvcc ~=13.3.73",
    "nvidia-cuda-cccl ~=13.3.3.4.1",
] + [
    spec.split(";")[0].strip() for spec in CUDA_RUNTIME_DEPENDENCIES
]

# $ORIGIN-relative RUNPATH baked into the installed extension so the
# dynamic loader finds the excluded CUDA libraries in their pip packages'
# site-packages directories at runtime, without LD_LIBRARY_PATH. From the
# extension at site-packages/cytnx/, the sibling packages are one level up:
# nvidia/cu13/lib, cutensor/lib, and cuquantum/lib (cutensornet-cu13 and
# custatevec-cu13 both install under cuquantum/). Each of those libraries in
# turn carries its own $ORIGIN-relative RUNPATH reaching across the three
# directories, so this only needs to cover the extension's own direct
# NEEDED entries; the transitive graph resolves itself. Semicolon-separated
# because CMAKE_INSTALL_RPATH is a CMake list (CMake emits it to the linker
# colon-separated).
CUDA_INSTALL_RPATH = ";".join(
    f"$ORIGIN/../{rel}"
    for rel in ("nvidia/cu13/lib", "cutensor/lib", "cuquantum/lib")
)

# SONAMEs provided by the pip packages above (including their transitive
# NVIDIA dependencies -- nvidia-cuda-nvrtc via nvidia-cublas,
# nvidia-nvjitlink via nvidia-cusolver), excluded from auditwheel's
# vendoring so the wheel relies on them at runtime instead of bundling
# its own copies. auditwheel preserves the CMAKE_INSTALL_RPATH above through
# `repair` (appending its own vendored-libs dir to it) precisely because
# these are excluded, so the loader keeps both search paths.
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
    skb["cmake"]["args"] = [
        "--preset=openblas-cuda",
        f"-DCMAKE_INSTALL_RPATH={CUDA_INSTALL_RPATH}",
        # Disable interprocedural optimization for the wheel build. On CUDA
        # this would emit -dlto, whose device link step (nvlink -dlink)
        # needs the offline device-LTO backend library -- which the pip
        # nvidia-cuda-nvcc wheel does not ship (it bundles only nvcc/
        # cudafe++/nvlink/ptxas/fatbinary), so nvlink aborts with
        # "elfLink linker library load error". CMakeLists.txt otherwise
        # defaults this ON; it honors an explicit -D override. IPO is a
        # per-target, not per-language, property, so this also drops host
        # -flto -- an accepted tradeoff for a working CUDA wheel until the
        # device-LTO backend is available in the pip toolchain.
        "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF",
    ]

    exclude_flags = " ".join(f"--exclude {name}" for name in EXCLUDED_SONAMES)
    linux = doc["tool"]["cibuildwheel"]["linux"]
    linux["repair-wheel-command"] = (
        f"auditwheel repair {exclude_flags} -w {{dest_dir}} {{wheel}}"
    )

    toolchain_args = " ".join(shlex.quote(spec) for spec in CUDA_BUILD_TOOLCHAIN)
    linux["before-all"] = (
        f"{linux['before-all']} && "
        f"bash ./tools/cibuildwheel_before_all_cuda.sh "
        f"{CUDA_TOOLCHAIN_PREFIX} {toolchain_args}"
    )

    # CMAKE_PREFIX_PATH may not be set yet on a checkout that predates the
    # conda-forge dependency migration (#1057) -- append rather than assume
    # it's there, so this keeps working regardless of merge order between
    # that PR and this one.
    existing_prefix_path = linux["environment"].get("CMAKE_PREFIX_PATH", "")
    new_prefix_path = (
        f"{existing_prefix_path};{CUDA_TOOLCHAIN_ROOT}"
        if existing_prefix_path
        else CUDA_TOOLCHAIN_ROOT
    )
    # Rebuild the inline table from scratch rather than assigning new keys
    # onto the parsed one in place: tomlkit does not add the separating
    # comma when a key is added to an already-parsed InlineTable, which
    # silently produces invalid TOML.
    env = tomlkit.inline_table()
    env.update(dict(linux["environment"]))
    env["CMAKE_PREFIX_PATH"] = new_prefix_path
    env["PATH"] = f"{CUDA_TOOLCHAIN_ROOT}/bin:$PATH"
    env["CMAKE_CUDA_COMPILER"] = f"{CUDA_TOOLCHAIN_ROOT}/bin/nvcc"
    env["CUTENSOR_ROOT"] = f"{CUDA_TOOLCHAIN_PREFIX}/cutensor"
    env["CUQUANTUM_ROOT"] = f"{CUDA_TOOLCHAIN_PREFIX}/cuquantum"
    linux["environment"] = env

    PYPROJECT.write_text(tomlkit.dumps(doc))


def main() -> None:
    doc = tomlkit.parse(PYPROJECT.read_text())
    rewrite_pyproject(doc)
    print("stamped pyproject.toml for cytnx-cuda")


if __name__ == "__main__":
    main()
