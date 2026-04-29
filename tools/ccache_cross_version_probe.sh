#!/usr/bin/env bash
set -euo pipefail

mkdir -p /project/.ccache-probe
cat > /project/.ccache-probe/probe.cpp <<'CPP'
int probe_sum(int x) { return x + 42; }
CPP

ccache g++ -c /project/.ccache-probe/probe.cpp -O2 -o /project/.ccache-probe/probe.o

build_tag="${CIBW_BUILD:-}"
if [[ -z "${build_tag}" ]]; then
  py_tag=$(python - <<'PYTAG'
import sys
print(f"cp{sys.version_info.major}{sys.version_info.minor}-manylinux_x86_64")
PYTAG
)
  build_tag="${py_tag}"
fi

stats_file="/host_ccache/ccache_stats_${build_tag}.txt"
debug_file="/host_ccache/ccache_debug_${build_tag}.txt"

ccache --print-stats > "${stats_file}"
{
  echo "CIBW_BUILD=${CIBW_BUILD:-}"
  echo "build_tag=${build_tag}"
  echo "CCACHE_DIR=${CCACHE_DIR:-}"
  echo "CCACHE_CONFIGPATH=${CCACHE_CONFIGPATH:-}"
  echo "CCACHE_COMPILERCHECK=${CCACHE_COMPILERCHECK:-}"
  echo "PWD=$(pwd)"
  echo "which g++: $(command -v g++)"
  echo "g++ --version:"; g++ --version || true
  echo "ccache --version:"; ccache --version || true
  echo "ccache --show-config:"; ccache --show-config || true
  echo "ccache --print-stats:"; ccache --print-stats || true
} > "${debug_file}" 2>&1
