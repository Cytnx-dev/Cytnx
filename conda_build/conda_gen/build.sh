start_time=$(date +%s)

# By default, setuptools creates separate working directories for each Python
# version. This prevents ccache from sharing cached objects between versions.
# Setting the working directory to "build" ensures ccache can reuse the cache
# across different Python versions.
$PYTHON -m pip install . -vv --no-deps --ignore-installed\
  --config-settings=--build-option=build_ext\
  --config-settings=--build-option=--build-temp=build

end_time=$(date +%s)
echo "Execution time: $((end_time - start_time)) seconds"
