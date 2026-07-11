#!/usr/bin/env bash
set -euo pipefail

# Build and optionally test a CMakePresets.json preset through one entry
# point. Builds use all available CPUs, keep RUN_BENCHMARKS and RUN_TESTS on
# (both checked to be a no-cost toggle -- see CLAUDE.md/build-test-workflow),
# and reuse whatever generator a build dir already has instead of
# reconfiguring on a mismatch.
#
# Usage:
#   build_preset.sh <preset> [--target <target>] [--test [args...]]
#
# <preset>        Any configurePreset name from CMakePresets.json, e.g.
#                 openblas-cpu, mkl-cpu, debug-openblas-cpu.
# --target <t>    CMake target to build. Defaults to `all`. `pycytnx` and
#                 `all` need Python bindings (BUILD_PYTHON=ON, pybind11
#                 discoverable); anything else (`test_main`, `cytnx`,
#                 `benchmarks_main`, ...) does not and skips the Python
#                 setup entirely.
# --test [args]   Run the target's tests after building. For a Python
#                 target, args are passed through to `pytest` verbatim (a
#                 path/`-k` filter fully replaces the default `pytests/`
#                 collection, matching normal pytest semantics); with no
#                 args, runs the full `pytests/` suite. For a non-Python
#                 target, a single optional arg is used as
#                 `--gtest_filter=<value>` against `tests/test_main`; with
#                 no arg, runs the full suite. `debug-*-cuda` presets get
#                 ASAN_OPTIONS set automatically (see below) -- no need to
#                 remember it.
#
# A Python target's first build for a preset (no venv yet at
# build/<preset>-venv) goes through `pip install --editable`, which is what
# makes pybind11 discoverable and sets up the import redirect pytest needs.
# Every later call for that preset -- Python target or not -- is a direct
# `cmake --build`, reusing that same build dir incrementally; no repeat pip
# overhead once the venv exists.

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <preset> [--target <target>] [--test [args...]]" >&2
  exit 1
fi

preset="$1"
shift

target="all"
do_test=0
test_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      target="$2"
      shift 2
      ;;
    --test)
      do_test=1
      shift
      test_args=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "${repo_root}"

build_dir="build/${preset}"
venv_dir="build/${preset}-venv"

case "${target}" in
  all | pycytnx) needs_python=1 ;;
  *) needs_python=0 ;;
esac

# Portable CPU count: nproc (Linux) or sysctl (macOS).
if command -v nproc >/dev/null 2>&1; then
  jobs="$(nproc)"
elif command -v sysctl >/dev/null 2>&1; then
  jobs="$(sysctl -n hw.ncpu)"
else
  jobs=4
fi

is_fresh_configure=0
if [[ ! -f "${build_dir}/CMakeCache.txt" ]]; then
  is_fresh_configure=1
fi

# Reuse an existing build dir's generator; never pass a conflicting -G, and
# never delete/reconfigure just because a mismatch might exist. Default a
# fresh dir to Ninja, matching pip/scikit-build-core's own default
# preference, so a Python and a non-Python build of the same fresh preset
# converge on the same generator without needing to agree in advance.
generator_args=()
if [[ "${is_fresh_configure}" -eq 1 ]] && command -v ninja >/dev/null 2>&1; then
  generator_args=(-G Ninja)
fi

if [[ ${needs_python} -eq 1 && ! -f "${venv_dir}/bin/activate" ]]; then
  # First Python build for this preset: pip sets up the venv, makes
  # pybind11 discoverable, and creates the editable-install import redirect.
  python3 -m venv "${venv_dir}"
  # shellcheck disable=SC1090
  source "${venv_dir}/bin/activate"
  pip install --upgrade pip --quiet

  # cmake.args is list-valued: repeated --config-settings=cmake.args=X
  # entries accumulate (verified empirically) rather than the last one
  # winning, so --preset and -G can both be passed this way. Pinning
  # --preset explicitly matters: pyproject.toml hardcodes
  # cmake.args=["--preset=openblas-cpu"], so without this override every
  # pip build would silently configure as openblas-cpu regardless of
  # <preset> (verified empirically).
  cmake_args=(--config-settings=cmake.args="--preset=${preset}")
  if [[ ${#generator_args[@]} -gt 0 ]]; then
    cmake_args+=(--config-settings=cmake.args=-GNinja)
  fi

  pip install --editable '.[dev]' \
    --config-settings=build-dir="${build_dir}" \
    "${cmake_args[@]}" \
    --config-settings=cmake.define.RUN_TESTS=ON \
    --config-settings=cmake.define.RUN_BENCHMARKS=ON \
    --config-settings=build.targets="${target}" \
    --config-settings=build.tool-args="-j${jobs}"
else
  configure_args=(-DRUN_TESTS=ON -DRUN_BENCHMARKS=ON)
  # BUILD_PYTHON=ON (the default) makes configure unconditionally require
  # pybind11, which is not installed outside a venv. Only a fresh,
  # non-Python configure needs the override -- an existing dir (e.g. one a
  # Python build already configured with BUILD_PYTHON=ON) is left as-is,
  # never reconfiguring a working dir's unrelated options.
  if [[ "${is_fresh_configure}" -eq 1 && ${needs_python} -eq 0 ]]; then
    configure_args+=(-DBUILD_PYTHON=OFF)
  fi
  if [[ "${is_fresh_configure}" -eq 1 ]]; then
    cmake --preset "${preset}" "${generator_args[@]}" "${configure_args[@]}"
  fi
  cmake --build "${build_dir}" --target "${target}" --parallel "${jobs}"
fi

if [[ ${do_test} -eq 0 ]]; then
  exit 0
fi

if [[ ${needs_python} -eq 1 ]]; then
  # shellcheck disable=SC1090
  source "${venv_dir}/bin/activate"
  if [[ ${#test_args[@]} -gt 0 ]]; then
    pytest "${test_args[@]}"
  else
    pytest pytests/
  fi
else
  binary="${build_dir}/tests/test_main"
  if [[ ! -x "${binary}" ]]; then
    echo "No test binary at ${binary} -- build --target test_main first." >&2
    exit 1
  fi
  # ASan only bites the debug+CUDA presets in practice; export the
  # workaround automatically instead of expecting the caller to remember it.
  case "${preset}" in
    debug-*-cuda)
      export ASAN_OPTIONS='protect_shadow_gap=0:replace_intrin=0:detect_leaks=0'
      ;;
  esac
  if [[ ${#test_args[@]} -gt 0 ]]; then
    "${binary}" "--gtest_filter=${test_args[0]}"
  else
    "${binary}"
  fi
fi
