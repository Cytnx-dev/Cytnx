#mkdir build
#cd build
#cmake ../ -DCMAKE_INSTALL_PREFIX=$PREFIX $SRC_DIR -DUSE_MKL=1
#make install -j4

# $PYTHON setup.py install     # Python command to install the script.
$PYTHON -m pip install . -vv --no-deps --no-build-isolation
