#!/usr/bin/env bash
set -euo pipefail

# Called by cibuildwheel AFTER the single CUDA wheel has been built.
# The host bind mount survives the manylinux container; no second Cytnx build.
output_dir="${CYTNX_BENCHMARK_EXPORT_DIR:?set CYTNX_BENCHMARK_EXPORT_DIR}"
build_dir="build/openblas-cuda"
test -f "${build_dir}/CMakeCache.txt" || {
  echo "Expected CMake build tree missing: ${build_dir}" >&2
  exit 1
}
test -f "${build_dir}/libcytnx.a" || {
  echo "Expected static Cytnx archive missing: ${build_dir}/libcytnx.a" >&2
  exit 1
}
mkdir -p "${output_dir}"
cmake --install "${build_dir}" --prefix "${output_dir}/install" --component Development
cmake --install "${build_dir}" --prefix "${output_dir}/install" --component headers
# Preserve exact source tree internal headers for the experimental benchmark.
cp -a src "${output_dir}/src"
cp -a benchmarks "${output_dir}/benchmarks"
cp -a tools/gpu_get_benchmark "${output_dir}/gpu_get_benchmark"
echo "Exported install tree to ${output_dir}"
