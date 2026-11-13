#!/usr/bin/env bash
# Reproducible native Kaggle CUDA Release L2 Google Benchmark.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive CUDA_VISIBLE_DEVICES=0
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
OUT="${CYTNX_GET_OUTPUT_DIR:-$ROOT/results/gpu_get}"
mkdir -p "$OUT"
exec > >(tee "$OUT/kaggle_run.log") 2>&1
echo "Starting $(date -u +%FT%TZ) commit $(git rev-parse HEAD)"
nvidia-smi
if ! command -v nvcc >/dev/null; then
  if [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export PATH="/usr/local/cuda/bin:$PATH"
  else echo "ERROR: Kaggle CUDA Toolkit does not contain nvcc"; exit 2; fi
fi
nvcc --version
python -m pip install -q 'cmake>=3.25,<4' ninja
if [[ "$(id -u)" == "0" ]]; then APT=(apt-get); else APT=(sudo apt-get); fi
"${APT[@]}" update -qq
"${APT[@]}" install -y -qq --no-install-recommends libboost-dev libopenblas-dev liblapacke-dev libarpack2-dev libbenchmark-dev
if [[ -x /usr/bin/g++-13 ]]; then
  export CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13
else export CC="${CC:-gcc}" CXX="${CXX:-g++}"; fi
if ! printf '#include <format>\nint main(){return std::format("{}",3).size()!=1;}\n' | "$CXX" -std=c++20 -x c++ - -o /tmp/cytnx-format-check 2>/dev/null; then
  echo "Installing GCC 13 for C++20 std::format"
  "${APT[@]}" install -y -qq --no-install-recommends gcc-13 g++-13
  export CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13
  printf '#include <format>\nint main(){return std::format("{}",3).size()!=1;}\n' | "$CXX" -std=c++20 -x c++ - -o /tmp/cytnx-format-check
fi
"$CXX" --version | head -1
ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits | head -1 | tr -d ' .\r')"
if [[ ! "$ARCH" =~ ^[0-9]+$ ]]; then echo "ERROR: cannot detect GPU compute capability"; exit 3; fi
echo "Detected compute capability = $ARCH"
NVCC="$(command -v nvcc)"
BUILD="$ROOT/build/kaggle-gpu-get"
cmake -S "$ROOT" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_CUDA_COMPILER="$NVCC" -DCMAKE_CUDA_HOST_COMPILER="$CXX" \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
  -DUSE_CUDA=ON -DRUN_BENCHMARKS=ON -DBUILD_PYTHON=OFF -DRUN_TESTS=OFF \
  -DUSE_DEBUG=OFF -DUSE_CUTENSOR=OFF -DUSE_CUQUANTUM=OFF -DUSE_HPTT=OFF \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$BUILD" --target benchmarks_main -j "${CYTNX_BUILD_JOBS:-2}"
EXE="$BUILD/benchmarks/benchmarks_main"
if [[ ! -x "$EXE" ]]; then echo "ERROR: benchmarks_main missing"; exit 4; fi
{
  echo "timestamp_utc=$(date -u +%FT%TZ)"
  echo "commit_sha=$(git rev-parse HEAD)"
  echo "branch=$(git branch --show-current)"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  echo "gpu_arch=$ARCH"
  echo "cmake_build_type=Release"
  echo "use_cuda=ON run_benchmarks=ON build_python=OFF run_tests=OFF"
  echo "use_cutensor=OFF use_cuquantum=OFF use_hptt=OFF"
  echo "lto=OFF coverage=OFF asan=OFF"
  "$CXX" --version | head -1
  nvcc --version
  nvidia-smi -L
} > "$OUT/environment.txt"
cp "$BUILD/CMakeCache.txt" "$OUT/CMakeCache.txt"
PILOT_FILTER='^GPUGetL2/(A262144/T262144/U1|B/T2048/U32|A262144/T256/U1024|B/T4/U65536)/Identity/S[0-3]($|/)'
"$EXE" --benchmark_filter="$PILOT_FILTER" --benchmark_out="$OUT/pilot.json" --benchmark_out_format=json 2>&1 | tee "$OUT/pilot.log"
python benchmarks/gpu/analyze_get_results.py pilot "$OUT/pilot.json" --output "$OUT/dispatch.txt"
POLICY="$(tr -d '\n\r' < "$OUT/dispatch.txt")"
export CYTNX_BENCH_GPU_GET_DISPATCH="$POLICY"
echo "Pilot-selected adaptive policy: $POLICY"
"$EXE" --benchmark_filter='^GPUGetL2/' --benchmark_out="$OUT/full.json" --benchmark_out_format=json 2>&1 | tee "$OUT/full.log"
python benchmarks/gpu/analyze_get_results.py report "$OUT/full.json" --output-dir "$OUT" --policy "$POLICY"
echo "DONE: $OUT/full.json $OUT/gpu_get_summary.csv $OUT/gpu_get_report.md"
