Install & Usage of Cytnx
============================
To install Cytnx, we recommend user to use anaconda/miniconda to install. However, advanced users can also build from source if necessary. 
Note that both python API and C++ API will be installed together regardless of which method you use. 


Conda install
********************
In the following we show how to install Cytnx with conda.

1. Install :anaconda:`Anaconda3 <>` / :miniconda:`Miniconda3 <>`.
    
    
2. Create a virtual environment:

* For linux/WSL:

.. code-block:: shell
    :linenos:

    $conda config --add channels conda-forge
    $conda create --channel conda-forge --name cytnx python=3.9 _openmp_mutex=*=*_llvm

.. Note::

    1. We do not support a native Windows package at this stage. If you are using Windows OS, please use WSL. 
    2. [0.9] Currently, the supported Python versions are updated to: linux --  3.8/3.9/3.10; MacOS-osx64 -- 3.7+ (no conda support). You can change the python=* argument to the version you like.  


* For MacOS:

    Please build from source, currently 0.9+ does not have conda package support.

..
    * For MacOS (non-arm arch)
    .. code-block:: shell
        :linenos:
        
        $conda config --add channels conda-forge
        $conda create --channel conda-forge --name cytnx python=3.8 llvm-openmp


.. note:: 

    * See :virtualenv:`This page <>` for how to use virtual environment in conda. 
    

3. Activate environment and conda install the cytnx package:
    
    Once you create a virtual environment, we need to activate the environment before starting to use it. 

.. code-block:: shell

    $conda activate cytnx
    
.. code-block:: shell
    
    $conda install -c kaihsinwu cytnx

.. note::

    * To install the GPU (CUDA) support version, use:

    $conda install -c kaihsinwu cytnx_cuda 


Once it is installed, we are all set, and ready to start using Cytnx. 


Using python API after Conda install
---------------------------------------
After installing Cytnx, using the Python API is very straight forward, simply import cytnx via:

* In Python:

.. code-block:: python 
    :linenos:

    import cytnx
    
    A = cytnx.ones(4)
    print(A)

Output>>

.. code-block:: text
    
    Total elem: 4
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (4)
    [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]
        
Using C++ API after Conda install
---------------------------------------
One important feature of Cytnx is that the installation also provides a C++ API. 
There are fundamental differences between C++ and Python, where C++ requires compiling and linking of the code, while Python as an interpreted language does not require both steps.

Cytnx provides a simple way for users to easily compile their C++ code. In the cytnx package, we provide three pre-set variables:

* In Python:

.. code-block:: python 
    :linenos:

    import cytnx
    cytnx.__cpp_include__
    cytnx.__cpp_lib__ 
    cytnx.__cpp_linkflags__
    cytnx.__cpp_flags__

* The first one **cytnx.__cpp_include__** gives you the cytnx header files directory path.
* The second one **cytnx.__cpp_lib__** gives you the cytnx library file directory path. 
* The third one **cytnx.__cpp_linkflags__** gives you the essential linking flags that are required when you link your own programs that using cytnx. 
* The fourth one **cytnx.__cpp_flags__** gives you the essential compiling flags that are required when you link your own programs that use cytnx. 

Let us consider the same example as before for Python. Here, we want to compile the **test.cpp** that uses Cytnx:

* test.cpp; in C++:

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


Now, to compile and link the above **test.cpp** which produces an executable **test**, we can simply use the following bash script:

.. code-block:: shell
    :linenos:

    export CYTNX_INC=$(python -c "exec(\"import cytnx\nprint(cytnx.__cpp_include__)\")")
    export CYTNX_LIB=$(python -c "exec(\"import cytnx\nprint(cytnx.__cpp_lib__)\")")/libcytnx.a
    export CYTNX_LINK="$(python -c "exec(\"import cytnx\nprint(cytnx.__cpp_linkflags__)\")")"
    export CYTNX_CXXFLAGS="$(python -c "exec(\"import cytnx\nprint(cytnx.__cpp_flags__)\")")"

    g++ -I${CYTNX_INC} ${CYTNX_CXXFLAGS} test.cpp ${CYTNX_LIB} ${CYTNX_LINK} -o test


The first four lines are the Python inline executions to get the three required attributes and store them into **CYTNX_INC**, **CYTNX_LIB**, **CYTNX_LINK** and **CYTNX_CXXFLAGS** variables. The last line is the usual compiling and linking of the C++ code **test.cpp**. After executing these steps, we can then run the program with the executable **test**. 


.. code-block:: shell
    
    ./test


Output>>

.. code-block:: text
    
    Total elem: 4
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (4)
    [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]


For users who what to use cmake/make to integrate Cytnx into more complex projects, one can use the following lines to extract the essential variables for cmake:

.. code-block:: shell

    CYTNX_INC := $(shell python -c "exec(\"import cytnx\nprint(cytnx.__cpp_include__)\")")
    CYTNX_LDFLAGS := $(shell python -c "exec(\"import cytnx\nprint(cytnx.__cpp_linkflags__)\")")
    CYTNX_LIB := $(shell python -c "exec(\"import cytnx\nprint(cytnx.__cpp_lib__)\")")/libcytnx.a
    CYTNX_CXXFLAGS := $(shell python -c "exec(\"import cytnx\nprint(cytnx.__cpp_flags__)\")")

    




.. toctree::

