set -xe

CYTNX_DEPS_PREFIX="/opt/cytnx-deps"

# Install required packages for manylinux_2_28+ (AlmaLinux/RHEL) from
# conda-forge instead of dnf. EPEL's arpack-devel links whichever OpenBLAS
# dnf happens to resolve, which is a pthreads-threaded build: it runs its
# own thread pool independent of the OpenMP runtime HPTT and cytnx's own
# OpenMP code use (-fopenmp, see CytnxBKNDCMakeLists.cmake), so a BLAS call
# made from inside an OpenMP region oversubscribes the CPU. conda-forge
# publishes an explicit "openmp" build variant of OpenBLAS that shares an
# OpenMP runtime's thread pool with the caller instead, closing that gap.
# ci-cmake_tests.yml's own mamba-based dependency install already relies on
# this same conda-forge OpenMP runtime for the native test build.
if command -v dnf >/dev/null 2>&1; then
    dnf install -y ccache curl bzip2

    arch="$(uname -m)"
    if [[ "${arch}" == "aarch64" || "${arch}" == "arm64" ]]; then
      conda_subdir="linux-aarch64"
    else
      conda_subdir="linux-64"
    fi
    curl -fLs "https://micro.mamba.pm/api/micromamba/${conda_subdir}/latest" | tar -xj -C /usr/local bin/micromamba
    MAMBA_ROOT_PREFIX=/opt/micromamba /usr/local/bin/micromamba create -y -p "${CYTNX_DEPS_PREFIX}" -c conda-forge \
      "openblas=*=*openmp*" \
      liblapacke \
      "arpack=*=nompi*" \
      libboost-headers

# musllinux_1_2 images are Alpine-based, so use apk there. conda-forge
# publishes no musl builds, so this path keeps installing from Alpine's own
# package index; the OpenBLAS-threading fix above only applies to the
# manylinux (dnf) wheels.
elif command -v apk >/dev/null 2>&1; then
    apk update
    apk add boost-dev openblas-dev arpack-dev ccache
else
    echo "Unsupported package manager: expected dnf or apk" >&2
    exit 1
fi
