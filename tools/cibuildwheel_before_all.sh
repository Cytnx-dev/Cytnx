set -xe
# Install ccache using pipx (available in manylinux images)
pipx install ccache

# Create symlinks for OpenBLAS headers if available
if [ -d /usr/include/openblas ]; then
    ln -sf /usr/include/openblas/lapacke.h /usr/include/lapacke.h
    ln -sf /usr/include/openblas/lapack.h /usr/include/lapack.h
    ln -sf /usr/include/openblas/lapacke_mangling.h /usr/include/lapacke_mangling.h
    ln -sf /usr/include/openblas/cblas.h /usr/include/cblas.h
    ln -sf /usr/include/openblas/openblas_config.h /usr/include/openblas_config.h
fi
