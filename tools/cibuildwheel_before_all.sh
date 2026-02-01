set -xe
# Use full path for dnf on AlmaLinux 8 based manylinux_2_28
/usr/bin/dnf install -y gcc-c++ glibc-devel kernel-headers
/usr/bin/dnf search lapack
/usr/bin/dnf --showduplicates list "lapack*"
/usr/bin/dnf install -y arpack-devel boost-devel ccache libomp-devel openblas-devel
ls /usr/include
ls /usr/include/openblas
ln -sf /usr/include/openblas/lapacke.h /usr/include/lapacke.h
ln -sf /usr/include/openblas/lapack.h /usr/include/lapack.h
ln -sf /usr/include/openblas/lapacke_mangling.h /usr/include/lapacke_mangling.h
ln -sf /usr/include/openblas/cblas.h /usr/include/cblas.h
ln -sf /usr/include/openblas/openblas_config.h /usr/include/openblas_config.h
ls -la /usr/include/cblas.h
ls -la /usr/include/openblas_config.h
