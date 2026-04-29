set -euxo pipefail

start_time=$(date +%s)

echo "Build diagnostics---------------------------------"
echo "uname: $(uname -a)"
echo "python: $($PYTHON --version 2>&1)"
echo "CC: ${CC:-<unset>}"
echo "CXX: ${CXX:-<unset>}"
echo "CMAKE_PRESET (from env): ${CMAKE_PRESET:-<unset>}"

# Conda-build selector expressions in meta.yaml can vary by platform metadata.
# Fallback here to avoid empty --preset values causing hard-to-debug failures.
if [[ -z "${CMAKE_PRESET:-}" ]]; then
  arch="$(uname -m)"
  case "$arch" in
    x86_64|amd64|i386|i686)
      CMAKE_PRESET="mkl-cpu"
      ;;
    *)
      CMAKE_PRESET="openblas-cpu"
      ;;
  esac
  echo "CMAKE_PRESET not provided by conda-build; using fallback preset: ${CMAKE_PRESET}"
fi

# By default, scikit-build-core creates separate working directories for each
# Python version. This prevents ccache from sharing cached objects between
# versions. Setting the working directory to "build" ensures ccache can reuse
# the cache across different Python versions.
# `--config-settings` overwrites the whole list setting, like
# `skbuild.cmake.args`, so the whole list setting have to be re-assigned even if
# only one value in the list is changed.
$PYTHON -m pip install . -vv --no-deps --ignore-installed \
  --config-settings skbuild.build-dir=build \
  --config-settings "skbuild.cmake.args=--preset=$CMAKE_PRESET" \
  --config-settings "skbuild.cmake.args=-G Unix Makefiles"

end_time=$(date +%s)
echo "Execution time: $((end_time - start_time)) seconds"
