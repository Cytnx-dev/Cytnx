#!/usr/bin/env bash
set -euo pipefail

# CMAKE_CXX_COMPILER_LAUNCHER / CMAKE_CXX_LINKER_LAUNCHER wrapper: CMake
# invokes this as `strip-coverage-launcher.sh <real-compiler-or-linker>
# <args...>`. cytnx's CMakeLists.txt applies --coverage via
# target_compile_options/target_link_options whenever RUN_TESTS=ON, and
# --coverage is a GCC/Clang driver shorthand that -fno-profile-arcs/
# -fno-test-coverage do not cancel (verified empirically: a trailing
# -fno-profile-arcs -fno-test-coverage still produced a .gcno file) --
# stripping the literal --coverage token before it reaches the real
# compiler/linker is what actually works.
real="$1"
shift

args=()
for a in "$@"; do
  if [[ "$a" == "--coverage" ]]; then
    continue
  fi
  args+=("$a")
done

exec "$real" "${args[@]}"
