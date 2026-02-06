set -xe

# Install required packages based on distro
if command -v apk &> /dev/null; then
    # musllinux (Alpine)
    apk add --no-cache boost-dev openblas-dev arpack-dev ccache
elif command -v dnf &> /dev/null; then
    # manylinux_2_28+ (AlmaLinux/RHEL)
    dnf install -y boost-devel openblas-devel arpack-devel ccache
elif command -v yum &> /dev/null; then
    # manylinux2014 (CentOS)
    yum install -y boost-devel openblas-devel arpack-devel ccache
else
    echo "WARNING: No package manager found"
fi

# Create symlinks for OpenBLAS headers if available
if [ -d /usr/include/openblas ]; then
    ln -sf /usr/include/openblas/lapacke.h /usr/include/lapacke.h
    ln -sf /usr/include/openblas/lapack.h /usr/include/lapack.h
    ln -sf /usr/include/openblas/lapacke_mangling.h /usr/include/lapacke_mangling.h
    ln -sf /usr/include/openblas/cblas.h /usr/include/cblas.h
    ln -sf /usr/include/openblas/openblas_config.h /usr/include/openblas_config.h
fi
