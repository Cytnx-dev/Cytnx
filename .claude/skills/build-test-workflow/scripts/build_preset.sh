#!/usr/bin/env bash
set -euo pipefail

# Build and optionally test a CMakePresets.json preset through one entry
# point. Builds use all available CPUs and reuse whatever generator a build
# dir already has instead of reconfiguring on a mismatch. An existing,
# non-fresh build dir is still reconfigured in place (a verified zero-cost
# toggle when nothing needs to change) so RUN_TESTS/RUN_BENCHMARKS actually
# match the requested target rather than whatever an earlier build in that
# dir left them at.
#
# RUN_TESTS is forced ON for every target except benchmarks_main. For
# --target benchmarks_main, RUN_TESTS is left untouched instead: it adds
# --coverage instrumentation to the cytnx library (see CMakeLists.txt),
# which benchmarks_main also links against, and that overhead would skew
# timings -- use a build dir that has never had RUN_TESTS turned on for
# uninstrumented benchmark numbers (see cross-revision-benchmark).
# RUN_BENCHMARKS is only turned on for --target benchmarks_main: the
# top-level CMakeLists.txt runs find_package(benchmark REQUIRED) whenever
# it's on, regardless of which target is actually being built, which would
# break the test_main/pycytnx workflow on a machine that never installed
# Google Benchmark (CI's own native-dependency list does not).
#
# Usage:
#   build_preset.sh <preset> [--target <target>] [--test [args...]]
#
# <preset>        Any configurePreset name from CMakePresets.json, e.g.
#                 openblas-cpu, mkl-cpu, debug-openblas-cpu.
# --target <t>    CMake target to build. Defaults to `all`. `pycytnx` and
#                 `all` need Python bindings (BUILD_PYTHON=ON, pybind11
#                 discoverable); anything else (`test_main`, `gpu_test_main`,
#                 `cytnx`, `benchmarks_main`, ...) does not and skips the
#                 Python setup entirely. May be a space-separated list, e.g.
#                 `--target "test_main gpu_test_main"` to build the CUDA
#                 suite's two binaries in one call.
# --test [args]   Run the target's tests after building.
#                 - Python target: args pass through to `pytest` verbatim (a
#                   path/`-k` filter fully replaces the default `pytests/`
#                   collection, matching normal pytest semantics, and is NOT
#                   combined with --doctest-modules -- pass it explicitly if
#                   needed); with no args, runs `pytest pytests/
#                   --doctest-modules`, matching CI.
#                 - `--target benchmarks_main`: args pass through to the
#                   Google Benchmark binary verbatim (e.g.
#                   `--benchmark_filter=<pattern>`); with no args, runs
#                   every registered benchmark. Never invoke this
#                   concurrently with another build or benchmark run -- CPU
#                   contention skews timings.
#                 - Any other target (`test_main`, `gpu_test_main`, or a
#                   space-separated combination of the two): runs through
#                   `ctest --test-dir <build_dir> --output-on-failure`,
#                   matching what CI runs rather than invoking the gtest
#                   binary directly. A single optional arg is used as
#                   `-R <value>` (a ctest *regex* against
#                   `ClassName.TestName`, not a gtest glob/`:`-joined
#                   filter); with no arg, runs every test discovered in the
#                   build dir. `debug-*-cuda` presets get ASAN_OPTIONS set
#                   automatically (see below) -- no need to remember it.
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

# git, not a BASH_SOURCE-relative path count: makes this resolve correctly
# even when the script has been copied elsewhere (e.g. cross-revision-benchmark
# copies it out before checking out revisions that predate its own existence
# in the tree), as long as the caller's cwd is inside the target repo.
repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

build_dir="build/${preset}"
venv_dir="build/${preset}-venv"

case "${target}" in
  all | pycytnx) needs_python=1 ;;
  *) needs_python=0 ;;
esac

# ASan only bites the debug+CUDA presets in practice; export the workaround
# before any build happens, not just before running tests. gtest_discover_tests
# defaults to POST_BUILD discovery, which runs the freshly built test binary
# as part of `cmake --build` itself to enumerate its cases -- so a debug-*-cuda
# binary that needs this workaround can already abort during the build step,
# before the script ever reaches the test-running code below.
case "${preset}" in
  debug-*-cuda)
    export ASAN_OPTIONS='protect_shadow_gap=0:replace_intrin=0:detect_leaks=0'
    ;;
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
    --config-settings=build.targets="${target}" \
    --config-settings=build.tool-args="-j${jobs}"
else
  configure_args=()
  if [[ "${target}" == "benchmarks_main" ]]; then
    # Never force RUN_TESTS here: it adds --coverage instrumentation to the
    # cytnx library (see CMakeLists.txt), which benchmarks_main also links
    # against, and that overhead would skew benchmark timings. Leave
    # whatever RUN_TESTS state the dir already has untouched -- use a build
    # dir that's never had RUN_TESTS turned on for uninstrumented numbers
    # (see cross-revision-benchmark). RUN_BENCHMARKS is the only flag this
    # target needs (and pays the find_package(benchmark REQUIRED) cost of).
    configure_args+=(-DRUN_BENCHMARKS=ON)
  else
    configure_args+=(-DRUN_TESTS=ON)
  fi
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
  else
    # Reconfigure in place so RUN_TESTS/RUN_BENCHMARKS actually reflect this
    # call's target, not just whatever an earlier build in this dir left
    # them at -- both are verified zero-cost toggles when nothing needs to
    # change, and required (not optional) when something does: an existing
    # dir that was never configured with RUN_TESTS=ON has no tests/
    # subdirectory at all, so `--target test_main` would otherwise fail with
    # an unknown-target error instead of building anything.
    cmake "${build_dir}" "${configure_args[@]}"
  fi
  # ${target} may be several space-separated names (e.g. "test_main
  # gpu_test_main" for the CUDA suite) -- word-split intentionally so each
  # becomes its own --target argument.
  # shellcheck disable=SC2206
  target_words=(${target})
  cmake --build "${build_dir}" --target "${target_words[@]}" --parallel "${jobs}"
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
    # Matches CI's own invocation (.github/workflows/ci-cmake_tests.yml);
    # without --doctest-modules a docstring regression would pass here and
    # only surface in CI.
    pytest pytests/ --doctest-modules
  fi
elif [[ "${target}" == "benchmarks_main" ]]; then
  # Google Benchmark's binary is not a ctest test (no add_test/
  # gtest_discover_tests registration) -- run it directly, args passed
  # straight through as its own CLI flags.
  binary="${build_dir}/benchmarks/benchmarks_main"
  if [[ ! -x "${binary}" ]]; then
    echo "No benchmark binary at ${binary} -- build --target benchmarks_main first." >&2
    exit 1
  fi
  "${binary}" "${test_args[@]}"
else
  # test_main/gpu_test_main register each gtest case as its own ctest test
  # via gtest_discover_tests, so ctest gives per-test output-on-failure and
  # matches what CI runs -- prefer it over invoking the gtest binary
  # directly. `--test-dir` (not `--preset`) because CMakePresets.json's
  # testPresets are hardcoded to only two configurePresets
  # (debug-openblas-cpu, debug-openblas-cuda) and don't generalize to every
  # preset this script accepts.
  # ASAN_OPTIONS for debug-*-cuda is already exported above, before the build.
  ctest_args=(--test-dir "${build_dir}" --output-on-failure --parallel "${jobs}")
  if [[ ${#test_args[@]} -gt 0 ]]; then
    ctest_args+=(-R "${test_args[0]}")
  fi
  ctest "${ctest_args[@]}"
fi
