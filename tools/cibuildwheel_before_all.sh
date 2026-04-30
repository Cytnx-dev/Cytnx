set -xe

# Install required packages for manylinux_2_28+ (AlmaLinux/RHEL).
if command -v dnf >/dev/null 2>&1; then
    dnf install -y boost-devel openblas-devel arpack-devel ccache
# musllinux_1_2 images are Alpine-based, so use apk there.
elif command -v apk >/dev/null 2>&1; then
    apk update
    apk add boost-dev openblas-dev arpack-dev ccache
else
    echo "Unsupported package manager: expected dnf or apk" >&2
    exit 1
fi

# Kept disabled by request since CMAKE_PREFIX_PATH already points to openblas.
# if [ -d /usr/include/openblas ]; then
#     ln -sf /usr/include/openblas/lapacke.h /usr/include/lapacke.h
#     ln -sf /usr/include/openblas/lapack.h /usr/include/lapack.h
#     ln -sf /usr/include/openblas/lapacke_mangling.h /usr/include/lapacke_mangling.h
#     ln -sf /usr/include/openblas/cblas.h /usr/include/cblas.h
#     ln -sf /usr/include/openblas/openblas_config.h /usr/include/openblas_config.h
#     if ls /usr/include/openblas/openblas_config-*.h >/dev/null 2>&1; then
#         for f in /usr/include/openblas/openblas_config-*.h; do
#             ln -sf "${f}" "/usr/include/$(basename "${f}")"
#         done
#     fi
# fi
