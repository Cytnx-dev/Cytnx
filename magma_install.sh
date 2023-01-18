
cd thirdparty/magma
echo "BACKEND = cuda\nFORT = true\nGPU_TARGET ?= Kepler Maxwell Pascal Volta Turing Ampere" > make.inc
make generate


## building from scratch 
## req: BLAS/OpenBlas or MKL 
##      cuda toolkit >= 11.8 
mkdir build
cd build
rm -rf *            # to clear any cached CMake configuration
cmake -DCMAKE_INSTALL_PREFIX=$HOME/MAGMA -DGPU_TARGET="Kepler Maxwell Pascal Volta Turing Ampere" ../
make lib -j2  # only build dense part!
#make install 


