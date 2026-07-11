#!/usr/bin/env bash
set -euo pipefail

# Build a CMakePresets.json preset either as a pip editable install (enables
# pytest) or as a direct CMake configure+build (gtest/ctest only, no wheel
# packaging). Both modes build with all available CPUs, keep RUN_BENCHMARKS
# and RUN_TESTS on (both are checked to be a no-cost toggle -- see
# CLAUDE.md/build-test-workflow), and reuse whatever generator a build dir
# already has instead of reconfiguring on a mismatch.
#
# Usage:
#   tools/build_preset.sh <preset> --pytest
#   tools/build_preset.sh <preset> --gtest-only
#
# <preset> is any configurePreset name from CMakePresets.json, e.g.
# openblas-cpu, mkl-cpu, debug-openblas-cpu.
#
# --pytest      One-time (or refresh) `pip install --editable` into a
#               dedicated venv at build/<preset>-venv, pinned to
#               build-dir=build/<preset>. Makes `pytest` available against
#               that preset. Safe to re-run; pip's editable install is
#               idempotent.
# --gtest-only  Direct `cmake --preset <preset>` + `cmake --build --target
#               test_main`. No venv, no Python wheel packaging -- the fast
#               path when only ctest/gtest is needed and the preset's build
#               dir already exists (e.g. was set up once via --pytest, or is
#               a debug preset a --pytest run never touched).
#
# Both modes reuse an existing build dir's generator (never pass -G against
# an already-configured dir) and default a FRESH dir to Ninja, so pip and
# direct cmake converge on the same generator instead of mixing Ninja and
# Unix Makefiles across the two entry points.

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <preset> --pytest|--gtest-only" >&2
  exit 1
fi

preset="$1"
mode="$2"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

build_dir="build/${preset}"
venv_dir="build/${preset}-venv"

# Portable CPU count: nproc (Linux) or sysctl (macOS). No GNU-only fallback
# needed since both are checked explicitly.
if command -v nproc >/dev/null 2>&1; then
  jobs="$(nproc)"
elif command -v sysctl >/dev/null 2>&1; then
  jobs="$(sysctl -n hw.ncpu)"
else
  jobs=4
fi

# If the build dir is already configured, its generator is authoritative --
# never pass a conflicting -G, and never delete/reconfigure just because a
# mismatch might exist. If it does not exist yet, default to Ninja so a
# fresh --gtest-only configure lines up with pip/scikit-build-core's own
# default preference (confirmed empirically: it picks Ninja whenever the
# `ninja` binary is on PATH).
is_fresh_configure=0
if [[ ! -f "${build_dir}/CMakeCache.txt" ]]; then
  is_fresh_configure=1
fi

generator_args=()
if [[ "${is_fresh_configure}" -eq 1 ]] && command -v ninja >/dev/null 2>&1; then
  generator_args=(-G Ninja)
fi

case "${mode}" in
  --pytest)
    if [[ ! -d "${venv_dir}" ]]; then
      python3 -m venv "${venv_dir}"
    fi
    # shellcheck disable=SC1090
    source "${venv_dir}/bin/activate"
    pip install --upgrade pip --quiet

    cmake_define_args=(
      --config-settings=cmake.define.RUN_TESTS=ON
      --config-settings=cmake.define.RUN_BENCHMARKS=ON
    )
    if [[ ${#generator_args[@]} -gt 0 ]]; then
      cmake_define_args+=(--config-settings=cmake.args=-GNinja)
    fi

    pip install --editable '.[dev]' \
      --config-settings=build-dir="${build_dir}" \
      "${cmake_define_args[@]}" \
      --config-settings=build.tool-args="-j${jobs}"
    ;;
  --gtest-only)
    configure_args=(-DRUN_TESTS=ON -DRUN_BENCHMARKS=ON)
    # BUILD_PYTHON=ON (the default) makes configure unconditionally require
    # pybind11, which is not installed outside a venv. Only this mode's
    # first, fresh configure needs the override -- an existing dir (e.g. one
    # a --pytest run already configured with BUILD_PYTHON=ON) is left as-is,
    # matching the "never reconfigure a working dir's unrelated options"
    # rule.
    if [[ "${is_fresh_configure}" -eq 1 ]]; then
      configure_args+=(-DBUILD_PYTHON=OFF)
    fi
    cmake --preset "${preset}" "${generator_args[@]}" "${configure_args[@]}"
    cmake --build "${build_dir}" --target test_main --parallel "${jobs}"
    ;;
  *)
    echo "Unknown mode: ${mode} (expected --pytest or --gtest-only)" >&2
    exit 1
    ;;
esac
