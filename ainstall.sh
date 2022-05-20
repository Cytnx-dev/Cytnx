rm -rf build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=/home/j9263178/cytnx_520/ -DUSE_MKL=on -DBUILD_PYTHON=on ../
make
make install