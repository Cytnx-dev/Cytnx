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
echo "Configuring standalone GPU Get benchmark"
cmake -S tools/gpu_get_benchmark -B "$output_dir/benchmark-build" -DCYTNX_SOURCE_DIR="$PWD" -DCMAKE_PREFIX_PATH="$output_dir/install" -DCMAKE_BUILD_TYPE=Release
cmake --build "$output_dir/benchmark-build" --target gpu_get_benchmark -j 2
mkdir -p "$output_dir/bin"
cp "$output_dir/benchmark-build/gpu_get_benchmark" "$output_dir/bin/"
echo "Exported benchmark executable and install tree from $build_dir"
