set -xe

# Install core development tools first
yum install -y gcc-c++ glibc-devel kernel-headers

yum search lapack
yum --showduplicates list "lapack*"
yum install -y arpack-devel boost-devel ccache libomp-devel openblas-devel

# On Red Hat/CentOS systems (which manylinux images are based on), the lapack-devel
# package typically provides the Fortran bindings and miss the C-interface header, so
# use headers shipped with openblas instead.
ls /usr/include
ls /usr/include/openblas

# Create symlinks for all required OpenBLAS headers
ln -sf /usr/include/openblas/lapacke.h /usr/include/lapacke.h
ln -sf /usr/include/openblas/lapack.h /usr/include/lapack.h
ln -sf /usr/include/openblas/lapacke_mangling.h /usr/include/lapacke_mangling.h
ln -sf /usr/include/openblas/cblas.h /usr/include/cblas.h
ln -sf /usr/include/openblas/openblas_config.h /usr/include/openblas_config.h

# Verify symlinks were created
ls -la /usr/include/cblas.h
ls -la /usr/include/openblas_config.h
