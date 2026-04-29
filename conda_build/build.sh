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

if ! "$PYTHON" -m pip --version; then
  echo "pip not found in build environment; collecting diagnostics." >&2
  "$PYTHON" - <<'PYDIAG'
import sys
print("sys.executable:", sys.executable)
print("sys.version:", sys.version)
try:
    import ensurepip
    print("ensurepip available:", ensurepip.__file__)
except Exception as exc:
    print("ensurepip unavailable:", repr(exc))
PYDIAG

  echo "Trying to bootstrap pip via ensurepip..." >&2
  if "$PYTHON" -m ensurepip --default-pip; then
    "$PYTHON" -m pip --version
  else
    echo "ERROR: failed to bootstrap pip with ensurepip." >&2
    exit 1
  fi
fi

PIP_VERSION_RAW=$("$PYTHON" - <<'PYPIPV'
from importlib.metadata import version
print(version("pip"))
PYPIPV
)
echo "pip version: ${PIP_VERSION_RAW}"
"$PYTHON" - <<'PYPIPCHK'
from importlib.metadata import version
from packaging.version import Version
if Version(version("pip")) < Version("23.1"):
    raise SystemExit("ERROR: pip>=23.1 is required so repeated --config-settings keys append correctly.")
PYPIPCHK

# By default, scikit-build-core creates separate working directories for each
# Python version. This prevents ccache from sharing cached objects between
# versions. Setting the working directory to "build" ensures ccache can reuse
# the cache across different Python versions.
# `--config-settings` overwrites the whole list setting, like
# `skbuild.cmake.args`, so the whole list setting have to be re-assigned even if
# only one value in the list is changed.
echo "scikit-build-core diagnostics--------------------------"
export SKBUILD_BUILD_DIR="build"
echo "Using skbuild.build-dir: ${SKBUILD_BUILD_DIR}"
echo "Using skbuild.cmake.args[0]: --preset=${CMAKE_PRESET}"
echo "Using skbuild.cmake.args[1]: -G Unix Makefiles"
"$PYTHON" -m pip show scikit-build-core || true
cmake --version || true

run_pip_install() {
  "$PYTHON" -m pip install . -vv --no-deps --ignore-installed --no-build-isolation \
    --config-settings "skbuild.build-dir=${SKBUILD_BUILD_DIR}" \
    --config-settings "skbuild.cmake.args=--preset=$CMAKE_PRESET" \
    --config-settings "skbuild.cmake.args=-G Unix Makefiles" \
    --config-settings "build.verbose=true"
}

if ! run_pip_install; then
  echo "ERROR: pip/scikit-build-core build failed. Collecting extra diagnostics..." >&2
  "$PYTHON" -m pip debug --verbose || true
  "$PYTHON" -m pip show scikit-build-core pybind11 cmake || true
  "$PYTHON" - <<'PYBUILDDBG'
import os
print("cwd:", os.getcwd())
print("CMAKE_PRESET:", os.environ.get("CMAKE_PRESET"))
print("skbuild build dir:", os.environ.get("SKBUILD_BUILD_DIR"))
print("skbuild build dir exists:", os.path.isdir(os.environ.get("SKBUILD_BUILD_DIR", "build")))
print("pyproject.toml exists:", os.path.isfile("pyproject.toml"))
PYBUILDDBG
  if [ -f CMakePresets.json ]; then
    echo "CMake presets file found:"
    sed -n '1,160p' CMakePresets.json || true
  fi
  exit 1
fi

end_time=$(date +%s)
echo "Execution time: $((end_time - start_time)) seconds"
