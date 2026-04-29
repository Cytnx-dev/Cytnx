start_time=$(date +%s)

# By default, scikit-build-core creates separate working directories for each
# Python version. This prevents ccache from sharing cached objects between
# versions. Setting the working directory to "build" ensures ccache can reuse
# the cache across different Python versions.
# `--config-settings` overwrites the whole list setting, like
# `skbuild.cmake.args`, so the whole list setting have to be re-assigned even if
# only one value in the list is changed.
install_log=$(mktemp)
if ! $PYTHON -m pip install . -vv --no-deps --ignore-installed \
  --config-settings skbuild.build-dir=build \
  --config-settings "skbuild.cmake.args=--preset=$CMAKE_PRESET" \
  --config-settings "skbuild.cmake.args=-G Unix Makefiles" 2>&1 | tee "$install_log"; then
  echo "pip install failed."
  echo "pip version: $($PYTHON -m pip --version 2>&1 || echo 'unavailable')"
  echo "Last 200 lines of pip install output:"
  tail -n 200 "$install_log"
  exit 1
fi

end_time=$(date +%s)
echo "Execution time: $((end_time - start_time)) seconds"
