set -xe

# Install OpenBLAS
# python -m pip install scipy-openblas64
# blas_lib_dir=$(python -c "import scipy_openblas64; print(scipy_openblas64.get_lib_dir())")
# blas_lib_name=$(python -c "import scipy_openblas64; print(scipy_openblas64.get_library())")
# # This handles different extensions (.so on Linux, .dylib on macOS)
# blas_lib_file=$(find "$blas_lib_dir" -name "*${blas_lib_name}.*" -type f | head -n 1)
# export BLAS_LIBRARIES="$blas_lib_file"

# # Verify BLAS_LIBRARIES points to an existing file
# if [ -f "$BLAS_LIBRARIES" ]; then
#     echo "✓ BLAS_LIBRARIES points to existing file: $BLAS_LIBRARIES"
# else
#     echo "✗ ERROR: BLAS_LIBRARIES does not point to a valid file: $BLAS_LIBRARIES"
#     exit 1
# fi

# BLA_SIZEOF_INTEGER=8 matches the ILP64 openblas64_ interface
# that scipy-openblas64 provides; this avoids CMake thinking
# it found a 32-bit BLAS.
# export BLA_SIZEOF_INTEGER=8


uname -s
# Set up Homebrew for Linux
# Homebrew was removed from PATH on Linux:
# https://github.com/actions/runner-images/issues/6283
if [[ "$(uname -s)" == "Linux" ]]; then
    # Find linuxbrew from root folder recursively
    # Common locations: /home/linuxbrew/.linuxbrew, /home/*/linuxbrew/.linuxbrew
    # brew_find=$(find / -name "*linuxbrew*" -type d 2>/dev/null)
    # linuxbrew_bin=$(echo $brew_find | head -n 1)
    # if [ -n "$linuxbrew_bin" ] && [ -d "$linuxbrew_bin/bin" ]; then
    #     echo "Found linuxbrew at: $linuxbrew_bin"
    #     export PATH="$linuxbrew_bin/bin:$PATH"
    # else
    #     echo "Warning: linuxbrew not found, trying default path"
    #     export PATH="/home/linuxbrew/.linuxbrew/bin:$PATH"
    # fi
    export PATH="/host/home/linuxbrew/.linuxbrew/bin:$PATH"
fi

# Install boost, ccache and arpack-ng
# All Ubuntu and macOS GitHub action runners have Homebrew installed.
brew install boost ccache arpack openblas

brew --prefix openblas
ls $(brew --prefix openblas)
brew --prefix ccache
ls $(brew --prefix ccache)
brew --prefix arpack
ls $(brew --prefix arpack)
brew --prefix boost
ls $(brew --prefix boost)
