#!/usr/bin/env bash
set -xe

# Installs the CUDA toolchain (nvcc, plus the runtime/dev components cytnx
# links against) into a self-contained prefix inside the manylinux
# container, so the build step can compile and link CUDA/cuTENSOR/cuQuantum
# code without a system-wide CUDA install. This mirrors the PyTorch-style
# runtime dependency approach tools/prepare_cuda_release.py uses for the
# wheel's own [project.dependencies] -- the same nvidia-*/cutensor-cu13/
# cuquantum packages, here installed at build time instead of at import
# time, and to an isolated --target directory rather than the build venv's
# site-packages so they don't leak into the wheel's own dependency
# resolution.
#
# The prefix and package specs are passed as arguments (not hardcoded here)
# so tools/prepare_cuda_release.py's CUDA_TOOLCHAIN_PREFIX /
# CUDA_BUILD_TOOLCHAIN stay the single source of truth for both.
#
# This runs after (chained by tools/prepare_cuda_release.py's before-all
# rewrite) tools/cibuildwheel_before_all.sh, which already installs
# arpack/openblas/boost -- cytnx-cuda still needs those for its non-GPU
# linear algebra paths.

toolchain_prefix="$1"
shift

python3 -m pip install --target "${toolchain_prefix}" "$@"

# nvidia-cuda-nvcc's binaries assume they can find their own siblings
# (cudafe++, nvlink, ptxas, ...) next to nvcc on PATH, which --target
# already gives them (all nvidia-* packages share the "nvidia" namespace
# package, so they merge into one nvidia/cu13/{bin,include,lib} tree).
chmod +x "${toolchain_prefix}/nvidia/cu13/bin/"*

# The pip CUDA wheels ship only the versioned sonames (libcudart.so.13,
# libcutensor.so.2, ...), but CMake's find_package(CUDAToolkit) and the
# cuTENSOR/cuQuantum finders resolve libraries with find_library(), which
# only matches the unversioned libX.so name. Create those dev symlinks in
# each toolchain lib dir so configuration can locate the import libraries.
# (CUTENSOR_ROOT/CUQUANTUM_ROOT point at the cutensor/ and cuquantum/
# namespace-package roots; nvidia-* all merge under nvidia/cu13/.)
for lib_dir in \
  "${toolchain_prefix}/nvidia/cu13/lib" \
  "${toolchain_prefix}/cutensor/lib" \
  "${toolchain_prefix}/cuquantum/lib"; do
  [ -d "${lib_dir}" ] || continue
  for versioned in "${lib_dir}"/lib*.so.*; do
    [ -e "${versioned}" ] || continue
    base="$(basename "${versioned}")"
    unversioned="${base%%.so.*}.so"
    ln -sf "${base}" "${lib_dir}/${unversioned}"
  done
done

echo "CUDA toolchain installed at ${toolchain_prefix}/nvidia/cu13"
"${toolchain_prefix}/nvidia/cu13/bin/nvcc" --version
