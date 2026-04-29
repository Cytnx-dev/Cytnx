start_time=$(date +%s)

# By default, scikit-build-core creates separate working directories for each
# Python version. This prevents ccache from sharing cached objects between
# versions. Setting the working directory to "build" ensures ccache can reuse
# the cache across different Python versions.
# `--config-settings` overwrites the whole list setting, like
# `skbuild.cmake.args`, so the whole list setting have to be re-assigned even if
# only one value in the list is changed.
install_log=$(mktemp)

ensure_pip_available() {
  if $PYTHON -m pip --version >/dev/null 2>&1; then
    return 0
  fi

  echo "pip is missing; attempting bootstrap with ensurepip..."
  if ! $PYTHON -m ensurepip --default-pip; then
    echo "Failed to bootstrap pip with ensurepip."
    return 1
  fi

  $PYTHON -m pip --version >/dev/null 2>&1
}

run_pip_install() {
  set -o pipefail
  $PYTHON -m pip install . -vv --no-deps --ignore-installed \
    --config-settings skbuild.build-dir=build \
    --config-settings "skbuild.cmake.args=--preset=$CMAKE_PRESET" \
    --config-settings "skbuild.cmake.args=-G Unix Makefiles" 2>&1 | tee "$install_log"
  local status=$?
  set +o pipefail
  return $status
}

if ! ensure_pip_available; then
  echo "pip install failed."
  echo "pip version: unavailable"
  exit 1
fi

if ! run_pip_install; then
  pip_version=$($PYTHON -m pip --version 2>/dev/null || echo "unavailable")
  echo "pip install failed."
  echo "pip version: ${pip_version}"
  echo "Last 200 lines of pip install output:"
  tail -n 200 "$install_log"
  exit 1
fi

end_time=$(date +%s)
echo "Execution time: $((end_time - start_time)) seconds"
