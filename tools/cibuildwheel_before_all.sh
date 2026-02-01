set -xe

# Install ccache from GitHub releases
CCACHE_VERSION=4.9.1
curl -L https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}-linux-x86_64.tar.xz | tar -xJ
cp ccache-${CCACHE_VERSION}-linux-x86_64/ccache /usr/local/bin/
ccache --version

# Install required packages
# manylinux_2_28 is AlmaLinux 8 based, try dnf first, then yum
if command -v dnf &> /dev/null; then
    dnf install -y boost-devel openblas-devel arpack-devel
elif command -v yum &> /dev/null; then
    yum install -y boost-devel openblas-devel arpack-devel
else
    echo "WARNING: No package manager found, assuming packages are pre-installed"
fi

# Create symlinks for OpenBLAS headers if available
if [ -d /usr/include/openblas ]; then
    ln -sf /usr/include/openblas/lapacke.h /usr/include/lapacke.h
    ln -sf /usr/include/openblas/lapack.h /usr/include/lapack.h
    ln -sf /usr/include/openblas/lapacke_mangling.h /usr/include/lapacke_mangling.h
    ln -sf /usr/include/openblas/cblas.h /usr/include/cblas.h
    ln -sf /usr/include/openblas/openblas_config.h /usr/include/openblas_config.h
fi
