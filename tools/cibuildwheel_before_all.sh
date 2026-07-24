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
# publishes no musl builds, so arpack/boost still come from Alpine's own
# package index -- but Alpine's openblas-dev is pthreads-threaded, same
# problem as EPEL's on manylinux (see above), and there's no prebuilt
# OpenMP alternative to swap in the way conda-forge provides one for
# manylinux. Build OpenBLAS from source instead, with USE_OPENMP=1, and
# replace Alpine's copy with it in place.
elif command -v apk >/dev/null 2>&1; then
    apk update
    apk add boost-dev openblas-dev arpack-dev ccache \
      build-base gfortran linux-headers perl git patchelf

    # arpack's own NEEDED entry names the exact soname it was linked
    # against (Alpine's OpenBLAS-provided ABI name, e.g. libopenblas.so.3);
    # discovering it from the installed libarpack.so directly, rather than
    # hardcoding a version, keeps this working if Alpine ever changes that
    # soname.
    arpack_lib="$(find /usr/lib -maxdepth 1 -name 'libarpack.so*' -type f | head -1)"
    test -n "${arpack_lib}" || { echo "libarpack.so not found under /usr/lib" >&2; exit 1; }
    openblas_soname="$(readelf -d "${arpack_lib}" | grep NEEDED | grep -o 'libopenblas\.so[^]]*' | head -1)"
    test -n "${openblas_soname}" || { echo "Could not determine the OpenBLAS soname arpack expects" >&2; exit 1; }

    openblas_prefix=/opt/openblas-omp
    git clone --depth 1 --branch v0.3.33 https://github.com/OpenMathLib/OpenBLAS.git /tmp/OpenBLAS
    make -C /tmp/OpenBLAS -j"$(nproc)" DYNAMIC_ARCH=1 USE_OPENMP=1 NUM_THREADS=256 NO_STATIC=1
    make -C /tmp/OpenBLAS install PREFIX="${openblas_prefix}"
    rm -rf /tmp/OpenBLAS

    openblas_real="$(readlink -f "${openblas_prefix}/lib/libopenblas.so")"
    patchelf --set-soname "${openblas_soname}" "${openblas_real}"

    # Replace Alpine's own OpenBLAS with the OpenMP build under the exact
    # soname arpack (and cytnx's own -lopenblas link line, via the
    # unversioned symlink) expect, so both resolve to the same, single,
    # OpenMP-threaded copy -- no CMAKE_PREFIX_PATH override needed, since
    # this replaces the files at the default system search paths.
    ln -sf "${openblas_real}" "/usr/lib/${openblas_soname}"
    ln -sf "${openblas_real}" /usr/lib/libopenblas.so
    ln -sf "${openblas_real}" /usr/lib/libblas.so
    ln -sf "${openblas_real}" /usr/lib/liblapack.so

    readelf -d "${openblas_real}" | grep -E 'SONAME|NEEDED'
    ldd "${arpack_lib}"
else
    echo "Unsupported package manager: expected dnf or apk" >&2
    exit 1
fi
