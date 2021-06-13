Install Cytnx
====================
To install cytnx, we recommend user to use anaconda/miniconda to install. However, advanced user can also build from soruce if nessasary. 
Note that both python API and C++ API will be installed together.

Conda install
******************
Following we show how to install cytnx from conda.

1. Install :anaconda:`Anaconda3 <>` / :miniconda:`Miniconda3 <>`.
    
    Note that cytnx currently support python 3.6 (py36), 3.7 (py37) and python 3.8 (py38). 
    

2. create a virtual enviroment:

.. code-block:: shell
    :linenos:

    $conda create --channel conda-forge --name cytnx python=3.7 _openmp_mutex=*=*_llvm


.. note:: 

    * See :virtualenv:`This page <>` for how to use virtual enviroment in conda. 
    * User can select the python version you want. we recommend using >=3.7

3. activate enviroment and conda install cytnx:
    
    Once you create a virtual enviroment, we need to activate enviroment before start using it. 

.. code-block:: shell

    $conda activate cytnx
    
.. code-block:: shell
    
    $conda install -c kaihsinwu cytnx

.. note::

    * to install the GPU (CUDA) support version, use:

    $conda install -c kaihsinwu cytnx_cuda 


Once it is installed, we are all set, and ready to start using cytnx. 


Using python API
******************
After install cytnx, using python API is very straight forward, simply import cytnx via:

.. code-block:: python 
    :linenos:

    import cytnx
    
    A = cytnx.ones(4)
    print(A)

* Output:

.. code-block:: text
    
    Total elem: 4
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (4)
    [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]
        
Using C++ API
******************
The most important feature is that Cytnx installation also provides C++ API. 
Since there are fundamental differents between C++ and python where C++ require compiling and linking of the code, while python as an interprete language does not require both steps.

Cytnx provides a simple way for user to easily compiling their C++ code. In cytnx package, we provides three pre-set variables:

.. code-block:: python 
    :linenos:

    import cytnx
    cytnx.__cpp_include__
    cytnx.__cpp_lib__ 
    cytnx.__cpp_linkflags__
    cytnx.__cpp_flags__

* The first one **cytnx.__cpp_include__** gives you the cytnx header files directory path.
* The second one **cytnx.__cpp_lib__** gives you the cytnx library file directory path. 
* The thrid one **cytnx.__cpp_linkflags__** gives you the essential linking flags that are required when you link your own programs that using cytnx. 
* The fourth one **cytnx.__cpp_flags__** gives you the essential compiling flags that are required when you link your own programs that using cytnx. 

Let's see the same simple example as aformentioned in python API. Here, we want to compile the **test.cpp** that using cytnx:

* test.cpp

.. code-block:: c++
    :linenos:

    #include "cytnx.hpp"
    #include <iostream>
    using namespace std;

    int main(){
        auto A = zeros(4);
        cout << A << endl;        
        return 0;
    }


Now, to compile and linking the above **test.cpp** to produce an executable **test**, we can simply use the following bash script:

.. code-block:: shell
    :linenos:

    export CYTNX_INC=$(python -c "exec(\"import cytnx\nprint(cytnx.__cpp_include__)\")")
    export CYTNX_LIB=$(python -c "exec(\"import cytnx\nprint(cytnx.__cpp_lib__)\")")
    export CYTNX_LINK="$(python -c "exec(\"import cytnx\nprint(cytnx.__cpp_linkflags__)\")")"
    export CYTNX_CXXFLAGS="$(python -c "exec(\"import cytnx\nprint(cytnx.__cpp_flags__)\")")"

    g++ -I${CYTNX_INC} ${CYTNX_CXXFLAGS} test.cpp ${CYTNX_LIB}/libcytnx.a ${CYTNX_LINK} -o test


The first four lines are the python inline execution to get the three attributes and store them into **CYTNX_INC**, **CYTNX_LIB**, **CYTNX_LINK** and **CYTNX_CXXFLAGS** variables. The last line is the standard simple compiling of the C++ code **test.cpp**. After execute these steps, we can then run this program with the executable **test**. 


.. code-block:: shell
    
    ./test


* Output:

.. code-block:: text
    
    Total elem: 4
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (4)
    [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]

Using C++ API with CMake
*************************
For user who what to use cmake/make to integrate cytnx into more complicated project, one can use the following lines to extract the essential to the cmake variables:

.. code-block:: shell

    CYTNX_INC := $(shell python -c "exec(\"import cytnx\nprint(cytnx.__cpp_include__)\")")
    CYTNX_LDFLAGS := $(shell python -c "exec(\"import cytnx\nprint(cytnx.__cpp_linkflags__)\")")
    CYTNX_LIB := $(shell python -c "exec(\"import cytnx\nprint(cytnx.__cpp_lib__)\")")/libcytnx.a
    CYTNX_CXXFLAGS := $(shell python -c "exec(\"import cytnx\nprint(cytnx.__cpp_flags__)\")")




In the case cytnx is installed locally from binary build, not from anaconda, one can use the following lines to extract the linking and compiling varialbes:

.. code-block:: shell

    CYTNX_INC := $(shell python -c "exec(\"import sys\nsys.path.append(\'$(CYTNX_ROOT)\')\nimport cytnx\nprint(cytnx.__cpp_include__)\")")
    CYTNX_LDFLAGS := $(shell python -c "exec(\"import sys\nsys.path.append(\'$(CYTNX_ROOT)\')\nimport cytnx\nprint(cytnx.__cpp_linkflags__)\")")
    CYTNX_LIB := $(shell python -c "exec(\"import sys\nsys.path.append(\'$(CYTNX_ROOT)\')\nimport cytnx\nprint(cytnx.__cpp_lib__)\")")/libcytnx.a
    CYTNX_CXXFLAGS := $(shell python -c "exec(\"import sys\nsys.path.append(\'$(CYTNX_ROOT)\')\nimport cytnx\nprint(cytnx.__cpp_flags__)\")")

.. Note::
    
    CYTNX_ROOT is the path where cytnx is installed from binary build. 
    

Build Cytnx from source
*************************
For advanced user who wish to build cytnx from source, we provides the cmake install. 


Dependencies
----------------
Cytnx required the following minimum dependencies:
    
* cmake >=3.14
* Boost v1.53+ [check_deleted, atomicadd, intrusive_ptr]
* openblas (lapacke)
* gcc v4.8.5+ (recommand v6+ or equivalent clang on Mac with C++11 support) (required -std=c++11)


In addition, you might want to install the following optional dependencies if you want cytnx to compile with features like openmp, mkl and/or CUDA support. 

[Openmp]

* openmp 

[MKL]

* intel mkl 

[CUDA]

* Nvidia cuda library v10+
* Nvidia cuDNN library 

[Python API]

* python >=3.6
* pybind11 
* python-graphviz 
* graphviz

.. note::

    For MacOS, standard brew install openblas will not work since it lack lapacke.h wrapper support. 

    If you are using MacOS, please install intel mkl (free) instead. 

    For python API, we recommend install of python using anaconda or miniconda

Using anaconda/conda for deps
------------------------------
We recommend using conda to handle all the dependency:

1. Install anaconda, setting the virtual enviroments

2. Install the following dependencies:

.. code-block:: shell

    $conda install numpy boost cmake pybind11 mkl mkl-inclde intel-openmp setuptools python-graphviz graphviz 


3. Make sure your system compiler has minimum -std=c++11 support.

4. in addition, if you want to have gpu support (compile with -DUSE_CUDA=on), then additional packages need to install:

.. code-block:: shell

    $conda install cudatoolkit
 
Using system for deps
-------------------------

**If you are using system packages not conda, the following libs/packages should be installed**
**We recommend using anaconda/conda for deps (see above)**

OS specific installation of minimum dependencies:

* Ubuntu:

.. code-block:: shell 

    $sudo apt-get install libboost-all-dev libopenblas-dev liblapack-dev liblapacke-dev cmake make curl g++ libomp-dev 

* MacOS:

1. install boost:    

.. code-block:: shell

    $brew install boost 


2. download and install intel mkl via :mkl-mac:`intel mkl <>`
    

* Windows:

.. note:: 

    For Windows user, please use :wsl:`WSL <>`. We recommend using ubuntu distribution, and follow the instruction of Ubuntu to install cytnx and dependencies. 



Compiling process [Auto set-up]
-----------------------------------
Starting from v0.6.4a, cytnx provide a easy step-by-step install tool:

.. code-block:: shell
    
    $python auto_install.py

This will prompt user through a series of install options, and generate a shell script **ainstall.sh**. To install, simply execute this script:

.. code-block:: shell

    $sh ainstall.sh




Compiling process [manually set-up]
-------------------------------------
Please see the following steps for the standard cmake compiling process and all the compiling options:


1. create a build directory:

.. code-block:: shell

    $make build
    $cd build

2. use cmake to auto matically generate compiling files:

.. code-block:: shell

    $cmake [option] <cytnx repo directory>

The following are the avaliable compiling option flags that you can specify in **[option]**:

+------------------------+-------------------+------------------------------------+
|       options          | default           |          description               |
+------------------------+-------------------+------------------------------------+
| -DCMAME_INSTALL_PREFIX | /usr/local/cytnx  | Install desination of the library  |
+------------------------+-------------------+------------------------------------+
| -DBUILD_PYTHON         |   ON              | Compile and install python API     |
+------------------------+-------------------+------------------------------------+
| -DUSE_MKL              |   OFF             | Compile Cytnx with intel MKL lib.  |
|                        |                   | If =off, default link to openblas  |
+------------------------+-------------------+------------------------------------+
| -DUSE_OMP              |   OFF             | Compile with openmp acceleration   |
|                        |                   | If USE_MKL=on, USE_OMP is forced=on|
+------------------------+-------------------+------------------------------------+
| -DUSE_CUDA             |   OFF             | Compile with CUDA GPU support      |
+------------------------+-------------------+------------------------------------+
| -DUSE_HPTT             |   OFF             | Accelerate tensor transpose with   |
|                        |                   | hptt                               |
+------------------------+-------------------+------------------------------------+

Additional options for HPTT if -DUSE_HPTT=on:

+-------------------------+-------------------+------------------------------------+
|       options           | default           |          description               |
+-------------------------+-------------------+------------------------------------+
| -DHPTT_ENABLE_FINE_TUNE |  OFF              | HPTT optimized with native hardware|
+-------------------------+-------------------+------------------------------------+
| -DHPTT_ENABLE_AVX       |  OFF              | Compile HPTT with AVX instruction  |
+-------------------------+-------------------+------------------------------------+
| -DHPTT_ENABLE_ARM       |  OFF              | Compile HPTT with ARM arch.        | 
+-------------------------+-------------------+------------------------------------+
| -DHPTT_ENABLE_IBM       |  OFF              | Compile HPTT with ppc64le arch     | 
+-------------------------+-------------------+------------------------------------+


3. compile the code:
    
    $make 

4. install to the target location:

    $make install 



.. toctree::

