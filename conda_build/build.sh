#!/usr/bin/env bash
set -euo pipefail

start_time=$(date +%s)

echo "[conda_build] Python: ${PYTHON:-<unset>}"
echo "[conda_build] CMAKE_PRESET (from conda selector): ${CMAKE_PRESET:-<unset>}"
echo "[conda_build] CC: ${CC:-<unset>}"
echo "[conda_build] CXX: ${CXX:-<unset>}"

if [[ -z "${CMAKE_PRESET:-}" ]]; then
  case "$(uname -m)" in
    x86_64|amd64)
      CMAKE_PRESET="mkl-cpu"
      ;;
    *)
      CMAKE_PRESET="openblas-cpu"
      ;;
  esac
  echo "[conda_build] CMAKE_PRESET was unset; using fallback preset: ${CMAKE_PRESET}"
fi

# By default, scikit-build-core creates separate working directories for each
# Python version. This prevents ccache from sharing cached objects between
# versions. Setting the working directory to "build" ensures ccache can reuse
# the cache across different Python versions.
#
# Note: `pip --config-settings` can be sensitive to duplicate keys depending on
# frontend/backend handling. Use CMAKE_ARGS for the generator and a single
# `skbuild.cmake.args` entry for the preset to avoid list-clobbering.
export CMAKE_ARGS="-G Unix Makefiles"

echo "[conda_build] Running pip build/install with preset: ${CMAKE_PRESET}"
"${PYTHON}" -m pip install . -vv --no-deps --ignore-installed \
  --config-settings skbuild.build-dir=build \
  --config-settings "skbuild.cmake.args=--preset=${CMAKE_PRESET}"

end_time=$(date +%s)
echo "Execution time: $((end_time - start_time)) seconds"
