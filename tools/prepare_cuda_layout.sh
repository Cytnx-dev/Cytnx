#!/usr/bin/env bash
# Repairs the layout of NVIDIA's CUDA pip wheels so CMake can configure
# against them, given the prefix the nvidia/, cutensor/ and cuquantum/
# namespace packages were installed under.
#
# Two consumers pass different prefixes: tools/cibuildwheel_before_all_cuda.sh
# installs the wheels into an isolated --target directory for the release
# build, while pixi.toml's Linux CUDA environment gets them in its own
# site-packages. The fixes are the same either way, and are idempotent.
set -xe

toolchain_prefix="$1"

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

# Device LTO (-dlto) makes nvlink dlopen libnvvm to run the offline LTO
# codegen. In a normal CUDA toolkit libnvvm lives at <top>/nvvm/lib64/;
# nvcc.profile's TOP-relative search and nvlink's default nvvmpath both look
# there. The nvidia-nvvm wheel instead relocates it to <top>/lib/libnvvm.so.4
# and leaves nvvm/ with only bin/ and libdevice/ (no lib64), so nvlink fails
# device LTO with "elfLink linker library load error". Recreate the canonical
# nvvm/lib64/libnvvm.so{,.4} so nvlink finds it there.
nvvm_lib64="${toolchain_prefix}/nvidia/cu13/nvvm/lib64"
libnvvm="$(find "${toolchain_prefix}/nvidia/cu13/lib" -maxdepth 1 -name 'libnvvm.so.*' | head -1)"
if [ -n "${libnvvm}" ]; then
  mkdir -p "${nvvm_lib64}"
  libnvvm_base="$(basename "${libnvvm}")"
  ln -sf "../../lib/${libnvvm_base}" "${nvvm_lib64}/${libnvvm_base}"
  ln -sf "${libnvvm_base}" "${nvvm_lib64}/libnvvm.so"
fi
