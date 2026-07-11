#!/usr/bin/env bash
set -euo pipefail

# Build and optionally test a CMakePresets.json preset through one entry
# point. Builds use all available CPUs and reuse whatever generator a build
# dir already has instead of reconfiguring on a mismatch. Only a fresh dir
# (no CMakeCache.txt yet) is ever configured -- a dir that already exists
# from a manual, non-script `cmake --preset` run keeps whatever flags it was
# given, including a missing RUN_TESTS/RUN_BENCHMARKS; always build a preset
# through this script the first time to get the flags below.
#
# A fresh build dir's first configure turns RUN_TESTS and RUN_BENCHMARKS ON
# unconditionally, regardless of --target: every dir this script creates can
# build test_main/gpu_test_main/benchmarks_main from then on with no later
# reconfigure, no matter which target the first call happened to ask for.
# This does mean Google Benchmark needs to be installed for ANY build
# through this script, not just a --target benchmarks_main one -- expected
# on an agent's own dev machine, unlike a minimal CI runner.
#
# RUN_TESTS=ON would normally add --coverage instrumentation to the cytnx
# library (see CMakeLists.txt), which would also skew benchmarks_main's
# timings since it links the same library -- CMAKE_CXX_COMPILER_LAUNCHER/
# CMAKE_CXX_LINKER_LAUNCHER point at strip-coverage-launcher.sh (alongside
# this script), which strips the literal --coverage token before it reaches
# the real compiler/linker. -fno-profile-arcs/-fno-test-coverage do NOT
# cancel --coverage (verified empirically -- a .gcno file is still produced
# either way), so every build through this script is uninstrumented by
# construction, not by asking CMakeLists.txt/CI to cooperate.
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
#                   collection, matching normal pytest semantics); with no
#                   args, runs the full `pytests/` suite.
#                 - `--target benchmarks_main`: args pass through to the
#                   Google Benchmark binary verbatim (e.g.
#                   `--benchmark_filter=<pattern>`); with no args, runs
#                   every registered benchmark. Never invoke this
#                   concurrently with another build or benchmark run -- CPU
#                   contention skews timings.
#                 - Any other target (`test_main`, `gpu_test_main`, or a
#                   space-separated combination of the two): runs through
#                   `ctest --test-dir <build_dir> --output-on-failure`
#                   instead of invoking the gtest binary directly, for
#                   per-test pass/fail output. Scoped to just the requested
#                   binary's tests (`-LE gpu`/`-L gpu`) when a CUDA preset's
#                   shared build dir has both test_main's and
#                   gpu_test_main's tests registered but only one was
#                   actually built by this call. A single optional arg is used as
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

# Resolved from repo_root, not the script's own location: a copy of this
# script (e.g. cross-revision-benchmark's edge-case note for a revision
# predating this script) would otherwise point CMake at a
# strip-coverage-launcher.sh that doesn't exist next to the copy -- the
# real repo checkout always has it at this fixed path.
coverage_launcher="${repo_root}/.claude/skills/build-test-workflow/scripts/strip-coverage-launcher.sh"

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

# A completion marker, not bin/activate's mere existence: `python3 -m venv`
# creates bin/activate immediately, before `pip install --editable` below
# ever runs, so a venv left behind by a prior failed install (missing native
# deps, a compile error) would otherwise look "done" and permanently skip
# the pip path on every retry, building/testing against whatever partial or
# stale state that failed attempt left in the build dir.
pip_install_done="${venv_dir}/.build_preset-pip-install-done"

if [[ ${needs_python} -eq 1 && ! -f "${pip_install_done}" ]]; then
  # First (or previously-failed) Python build for this preset: pip sets up
  # the venv, makes pybind11 discoverable, and creates the editable-install
  # import redirect.
  python3 -m venv "${venv_dir}"
  # shellcheck disable=SC1090
  source "${venv_dir}/bin/activate"
  pip install --upgrade pip --quiet

  # cmake.args is list-valued: repeated --config-settings=cmake.args=X
  # entries accumulate (verified empirically) rather than the last one
  # winning, so --preset, -G, and the launcher vars can all be passed this
  # way. Pinning --preset explicitly matters: pyproject.toml hardcodes
  # cmake.args=["--preset=openblas-cpu"], so without this override every
  # pip build would silently configure as openblas-cpu regardless of
  # <preset> (verified empirically).
  cmake_args=(--config-settings=cmake.args="--preset=${preset}")
  if [[ ${#generator_args[@]} -gt 0 ]]; then
    cmake_args+=(--config-settings=cmake.args=-GNinja)
  fi
  if [[ "${is_fresh_configure}" -eq 1 ]]; then
    cmake_args+=(
      --config-settings=cmake.args="-DCMAKE_CXX_COMPILER_LAUNCHER=${coverage_launcher}"
      --config-settings=cmake.args="-DCMAKE_CXX_LINKER_LAUNCHER=${coverage_launcher}"
    )
  fi

  pip install --editable '.[dev]' \
    --config-settings=build-dir="${build_dir}" \
    "${cmake_args[@]}" \
    --config-settings=cmake.define.RUN_TESTS=ON \
    --config-settings=cmake.define.RUN_BENCHMARKS=ON \
    --config-settings=build.targets="${target}" \
    --config-settings=build.tool-args="-j${jobs}"
  # Only reached if the pip install above succeeded (set -e exits the
  # script on failure before this line runs), so a later retry correctly
  # re-enters this branch instead of trusting an incomplete venv.
  touch "${pip_install_done}"
else
  if [[ "${is_fresh_configure}" -eq 1 ]]; then
    configure_args=(-DRUN_TESTS=ON -DRUN_BENCHMARKS=ON
      -DCMAKE_CXX_COMPILER_LAUNCHER="${coverage_launcher}"
      -DCMAKE_CXX_LINKER_LAUNCHER="${coverage_launcher}")
    # BUILD_PYTHON=ON (the default) makes configure unconditionally require
    # pybind11, which is not installed outside a venv. Only a fresh,
    # non-Python configure needs the override -- an existing dir (e.g. one a
    # Python build already configured with BUILD_PYTHON=ON) is left as-is,
    # never reconfiguring a working dir's unrelated options.
    if [[ ${needs_python} -eq 0 ]]; then
      configure_args+=(-DBUILD_PYTHON=OFF)
    fi
    cmake --preset "${preset}" "${generator_args[@]}" "${configure_args[@]}"
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
    pytest pytests/
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
  # via gtest_discover_tests, so ctest gives per-test output-on-failure --
  # prefer it over invoking the gtest binary directly. `--test-dir` (not
  # `--preset`) because CMakePresets.json's
  # testPresets are hardcoded to only two configurePresets
  # (debug-openblas-cpu, debug-openblas-cuda) and don't generalize to every
  # preset this script accepts.
  # ASAN_OPTIONS for debug-*-cuda is already exported above, before the build.
  ctest_args=(--test-dir "${build_dir}" --output-on-failure --parallel "${jobs}")
  # A CUDA preset's shared build dir registers both test_main's and
  # gpu_test_main's tests once RUN_TESTS=ON, regardless of which one this
  # call actually built -- --test-dir alone doesn't scope to just the
  # requested target, so an unbuilt sibling binary's tests would otherwise
  # report as NOT_BUILT (or, if it happens to be already built, run too).
  # ${target_words[@]} is already set by the build step above.
  has_test_main=0
  has_gpu_test_main=0
  for word in "${target_words[@]}"; do
    case "${word}" in
      test_main) has_test_main=1 ;;
      gpu_test_main) has_gpu_test_main=1 ;;
    esac
  done
  if [[ ${has_test_main} -eq 1 && ${has_gpu_test_main} -eq 0 ]]; then
    ctest_args+=(-LE gpu)
  elif [[ ${has_gpu_test_main} -eq 1 && ${has_test_main} -eq 0 ]]; then
    ctest_args+=(-L gpu)
  fi
  if [[ ${#test_args[@]} -gt 0 ]]; then
    ctest_args+=(-R "${test_args[0]}")
  fi
  ctest "${ctest_args[@]}"
fi
