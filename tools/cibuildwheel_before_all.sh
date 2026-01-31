set -xe

yum search lapack
yum --showduplicates list "lapack*"
# yum install -y arpack-devel boost-devel ccache lapack-devel libomp-devel openblas-devel
yum install -y arpack-devel boost-devel ccache libomp-devel openblas-devel
# On Red Hat/CentOS systems (which manylinux images are based on), the lapack-devel
# package typically provides the Fortran bindings and miss the C-interface header, so
# use lapacke.h shipped with openblas instead.
ls /usr/include
ln -s /usr/include/openblas/lapacke.h /usr/include/lapacke.h
ln -s /usr/include/openblas/lapack.h /usr/include/lapack.h
ln -s /usr/include/openblas/lapacke_mangling.h /usr/include/lapacke_mangling.h
ln -s /usr/include/openblas/cblas.h /usr/include/cblas.h
ln -s /usr/include/openblas/openblas_config.h /usr/include/openblas_config.h
ls /usr/include/openblas
#  boost1.78-devel
# ls /usr/include/boost | head
# ls /usr/lib64/libboost_*.so
