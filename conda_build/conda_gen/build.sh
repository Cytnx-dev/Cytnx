#mkdir build
#cd build
#cmake ../ -DCMAKE_INSTALL_PREFIX=$PREFIX $SRC_DIR -DUSE_MKL=1
#make install -j4

# $PYTHON setup.py install     # Python command to install the script.

# By default, setuptools creates separate working directories for each Python
# version. This prevents ccache from sharing cached objects between versions.
# Setting the working directory to "build" ensures ccache can reuse the cache
# across different Python versions.
$PYTHON -m pip install . -vv --no-deps --no-build-isolation\
  --config-settings=--build-option=build_ext\
  --config-settings=--build-option=--build-temp=build
