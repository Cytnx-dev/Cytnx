# For developers

- we enforce clang-format for the c++ code. Setting up the pre-commit tools for auto formatting:
  1. install [pre-commit](https://pre-commit.com/)
  2. at repo directory, run

     $ pre-commit install

     to set up the git hook scripts


- for developing torch backend:
  1. Install pytorch using mamba (or conda) install (https://pytorch.org/)
  2. mamba (or conda) install cmake make boost libboost git compilers pybind11
     [Note] we let pytorch to handle all the linalg deps

  3. set BACKEND_TORCH=ON in Install.sh, this will ignore all the rest of variables.


- for developing cytnx backend:
  -> follow user guide





- for general develop:

  1. copy dev_test.cpp to repo root directory,
     This will serve as your scratch pad for experiment on the code base.

  2. add option to cmake `-DDEV_MODE=on` will build `dev_test.cpp` linking with cytnx.
     After make, an dev_test executable should appears on repo root directory (no `make install` needed)



## note:
1. cmake 3.27.0 changed the FindPythonInterp / FindPythonLib, so install cmake=3.26 instead for now
2. 3.11 will fail to interact with pybind11, so for now support is up to 3.10
