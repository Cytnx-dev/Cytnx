set -xe

# Install ccache from sccache (static binary) as alternative
# Or try older ccache version that might be more compatible
CCACHE_VERSION=4.8.3
curl -L https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}-linux-x86_64.tar.xz | tar -xJ
cp ccache-${CCACHE_VERSION}-linux-x86_64/ccache /usr/local/bin/
chmod +x /usr/local/bin/ccache
# Check if it works, if not, create a dummy ccache that just passes through
if ! /usr/local/bin/ccache --version 2>/dev/null; then
    echo '#!/bin/bash' > /usr/local/bin/ccache
    echo 'exec "${@}"' >> /usr/local/bin/ccache
    chmod +x /usr/local/bin/ccache
    echo "WARNING: Using dummy ccache passthrough"
fi
ccache --version || echo "ccache passthrough mode"

# Install required packages
if command -v dnf &> /dev/null; then
    dnf install -y boost-devel openblas-devel arpack-devel
elif command -v yum &> /dev/null; then
    yum install -y boost-devel openblas-devel arpack-devel
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
