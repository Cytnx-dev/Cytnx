#rm -rf build
#mkdir build
cd build
#cmake -DCMAKE_INSTALL_PREFIX=/home/petjelinux/Cytnx_lib -DUSE_MKL=on -DUSE_HPTT=on -DHPTT_ENABLE_FINE_TUNE=on -DHPTT_ENABLE_AVX=on -DBUILD_PYTHON=on -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DRUN_TESTS=on ../
make -j `nproc`
make install
GTEST_COLOR=1 ctest -V --output-junit junit.xml
