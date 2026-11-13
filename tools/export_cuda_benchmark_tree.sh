#!/usr/bin/env bash
set -euo pipefail
output_dir="${CYTNX_BENCHMARK_EXPORT_DIR:?}"
mkdir -p "$output_dir"
echo "Searching for scikit-build-core CMake cache:"
find build -name CMakeCache.txt -print
cache="$(find build -name CMakeCache.txt -print -quit)"
test -n "$cache" || { echo "No CMakeCache.txt found" >&2; exit 1; }
build_dir="$(dirname "$cache")"
cmake --install "$build_dir" --prefix "$output_dir/install"
cp -a src "$output_dir/src"
cp -a benchmarks "$output_dir/benchmarks"
cp -a tools/gpu_get_benchmark "$output_dir/gpu_get_benchmark"
echo "Exported Cytnx install tree and benchmark sources from $build_dir"
