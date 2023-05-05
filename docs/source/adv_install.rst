Advanced Install of Cytnx
============================

Build/Install Cytnx from source
*********************************
For advanced user who wish to build Cytnx from source, we provide cmake install. 


Dependencies
-------------------
Cytnx requires the following minimum dependencies:
    
* cmake >=3.14
* git 
* make
* Boost v1.53+ [check_deleted, atomicadd, intrusive_ptr] 
* openblas (or mkl, see below) 
* gcc v6+ (or icpc, see below) (recommand latest or equivalent clang on Mac/Linux with C++11 support) (required -std=c++11)

In addition, you might want to install the following optional dependencies if you want Cytnx to compile with features like openmp, mkl and/or CUDA support. 

[Openmp]

* openmp

[MKL]

* intel mkl 

[CUDA]

* Nvidia cuda library v10+
* Nvidia cuDNN library 
* Nvidia cuTensor library
* Nvidia cuQuantum library

[MAGMA]

* MAGMA 2.7 currently required strict cuda version v11.8


[Python API]

* python >= 3.6
* pybind11 
* python-graphviz 
* graphviz
* numpy 

There are two methods how you can set-up all the dependencies before starting the build process: 

1. Using conda to install dependencies 
2. Directly install dependencies one-by-one via system package manager 

.. Note::
    
    We recommend using conda to handle all the dependencies (including compiling tools). 
    This is the simplest way as conda automatically resolves the whole path of each dependency, allowing cmake to automatically capture those.  

    How it works?
        
        >> The conda-forge channel includes not only the Python package but also other pre-compiled libraries/compilers. 



**Option A. Using anaconda/conda to install deps:**

1. Install anaconda/miniconda, setting the virtual environments

* For Linux/WSL:

.. code-block:: shell

    $conda config --add channels conda-forge
    $conda create --name cytnx python=3.8 _openmp_mutex=*=*_llvm
    $conda activate cytnx
    $conda upgrade --all 
    

* For MacOS:

.. code-block:: shell

    $conda config --add channels conda-forge
    $conda create --name cytnx python=3.8 llvm-openmp
    $conda activate cytnx
    $conda upgrade --all 

.. Note::
    
    1. The python=3.8 indicates the Python version you want to use. Generally, Cytnx is tested with 3.7/3.8/3.9. You can replace this with the version you want to use. 
    2. The last line is updating all the libraries such that they are all dependent on the conda-forge channel. 


2. Install the following dependencies:

.. code-block:: shell

    $conda install cmake make boost libboost git compilers numpy mkl mkl-include mkl-service pybind11 libblas=*=*mkl


.. Note:: 

    This installation includes the compilers/linalg libraries provided by conda-forge, so the installation of compilers on system side is not required. 

.. Hint::

    Trouble shooting:

        1. Make sure **conda-forge** channel has the top priority
        2. Make sure libblas=mkl (you can check using *conda list | grep libblas*) 


4. In addition, if you want to have GPU support (compile with -DUSE_CUDA=on), then additional packages need to be installed:

.. .. code-block:: shell

..     $conda install cudatoolkit cudatoolkit-dev 

.. code-block:: shell

    $conda conda install -c nvidia cuda


**Option B. Install deps via system package manager**

You can also choose to install deps directly from the system package manager, but one needs to carefully resolve the dependency path for cmake to capture them successfully. 


.. warning::

    For MacOS, standard brew install openblas will not work since it lacks lapacke.h wrapper support. 

    If you are using MacOS, please install intel mkl (free) instead. 

    For the Python API, we recommend installing Python using anaconda or miniconda.


 
Compiling process 
-------------------
Once you installed all the dependencies, it is time to start building the Cytnx source code. 

Starting from v0.7.6a, Cytnx provides a shell script **Install.sh**, which contains all the cmake arguments check list. To install, un-comment and put custom parameters on the corresponding lines, and simply execute this script:

.. code-block:: shell

    $sh Install.sh


**Arguments/options for cmake install**

Please see the following steps for the standard cmake compiling process and all the compiling options:


1. Create a build directory:

.. code-block:: shell

    $make build
    $cd build

2. Use cmake to automatically generate compiling files:

.. code-block:: shell

    $cmake [option] <Cytnx repo directory>

The following are the available compiling option flags that you can specify in **[option]**:

+------------------------+-------------------+------------------------------------+
|       options          | default           |          description               |
+------------------------+-------------------+------------------------------------+
| -DCMAME_INSTALL_PREFIX | /usr/local/cytnx  | Install destination of the library |
+------------------------+-------------------+------------------------------------+
| -DBUILD_PYTHON         |   ON              | Compile and install Python API     |
+------------------------+-------------------+------------------------------------+
| -DUSE_ICPC             |   OFF             | Compile using intel icpc compiler  |
+------------------------+-------------------+------------------------------------+
| -DUSE_MKL              |   OFF             | Compile Cytnx with intel MKL lib.  |
|                        |                   | If =off, default link to openblas  |
+------------------------+-------------------+------------------------------------+
| -DUSE_OMP              |   ON              | Compile with openmp acceleration   |
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


3. Compile the code:
    
    $make 

4. Install to the target location:

    $make install 



Using Python API after self-build install
-------------------------------------------
To use the Python API after self-build, you need to add the path where you installed Cytnx before importing it. 
The simplest (and most flexible) way to do that is to add it into sys.path right at the beginning of your code. 

In the following, we will use **CYTNX_ROOT** (capital letters) to represent the path where you installed Cytnx. You should replace it with the path where Cytnx is installed. 

.. code-block:: python 
    :linenos:

    import sys
    sys.path.insert(0,CYTNX_ROOT)
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


Using C++ API after self-build install
------------------------------------------
In the case that Cytnx is installed locally from binary build, not from anaconda, one can use the following lines to extract the linking and compiling variables:

.. code-block:: shell

    CYTNX_INC := $(shell python -c "exec(\"import sys\nsys.path.append(\'$(CYTNX_ROOT)\')\nimport cytnx\nprint(cytnx.__cpp_include__)\")")
    CYTNX_LDFLAGS := $(shell python -c "exec(\"import sys\nsys.path.append(\'$(CYTNX_ROOT)\')\nimport cytnx\nprint(cytnx.__cpp_linkflags__)\")")
    CYTNX_LIB := $(shell python -c "exec(\"import sys\nsys.path.append(\'$(CYTNX_ROOT)\')\nimport cytnx\nprint(cytnx.__cpp_lib__)\")")/libcytnx.a
    CYTNX_CXXFLAGS := $(shell python -c "exec(\"import sys\nsys.path.append(\'$(CYTNX_ROOT)\')\nimport cytnx\nprint(cytnx.__cpp_flags__)\")")

.. Note::
    
    CYTNX_ROOT is the path where Cytnx is installed from binary build. 



.. toctree::

