#!/usr/bin/env bash
set -euo pipefail

mkdir -p /project/.ccache-probe
cat > /project/.ccache-probe/probe.cpp <<'CPP'
int probe_sum(int x) { return x + 42; }
CPP

ccache g++ -c /project/.ccache-probe/probe.cpp -O2 -o /project/.ccache-probe/probe.o
ccache --print-stats > "/host_ccache/ccache_stats_${CIBW_BUILD}.txt"
