Install & Usage of Cytnx
============================
To install cytnx, we recommend user to use anaconda/miniconda to install. However, advanced user can also build from soruce if nessasary. 
Note that both python API and C++ API will be installed together regardless of which method you use. 


Conda install
********************
Following we show how to install cytnx from conda.

1. Install :anaconda:`Anaconda3 <>` / :miniconda:`Miniconda3 <>`.
    
    
2. create a virtual enviroment:

* For linux/WSL:

.. code-block:: shell
    :linenos:

    $conda config --add channels conda-forge
    $conda create --channel conda-forge --name cytnx python=3.9 _openmp_mutex=*=*_llvm

.. Note::

    1. We do not support native Windows package at this stage. if you are using Windows OS, please use WSL. 
    2. [0.7.6+] Currently, supporting python versions are updated to linux: 3.7/3.8/3.9; MacOS-osx64 3.7/3.8. You can change python=* argument to the version you want.  


* For MacOS (non-arm arch)

.. code-block:: shell
    :linenos:
    
    $conda config --add channels conda-forge
    $conda create --channel conda-forge --name cytnx python=3.8 llvm-openmp

.. note:: 

    * See :virtualenv:`This page <>` for how to use virtual enviroment in conda. 
    

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


Using python API after Conda install
---------------------------------------
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
        
Using C++ API after Conda install
---------------------------------------
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
    export CYTNX_LIB=$(python -c "exec(\"import cytnx\nprint(cytnx.__cpp_lib__)\")")/libcytnx.a
    export CYTNX_LINK="$(python -c "exec(\"import cytnx\nprint(cytnx.__cpp_linkflags__)\")")"
    export CYTNX_CXXFLAGS="$(python -c "exec(\"import cytnx\nprint(cytnx.__cpp_flags__)\")")"

    g++ -I${CYTNX_INC} ${CYTNX_CXXFLAGS} test.cpp ${CYTNX_LIB} ${CYTNX_LINK} -o test


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


For user who what to use cmake/make to integrate cytnx into more complicated project, one can use the following lines to extract the essential to the cmake variables:

.. code-block:: shell

    CYTNX_INC := $(shell python -c "exec(\"import cytnx\nprint(cytnx.__cpp_include__)\")")
    CYTNX_LDFLAGS := $(shell python -c "exec(\"import cytnx\nprint(cytnx.__cpp_linkflags__)\")")
    CYTNX_LIB := $(shell python -c "exec(\"import cytnx\nprint(cytnx.__cpp_lib__)\")")/libcytnx.a
    CYTNX_CXXFLAGS := $(shell python -c "exec(\"import cytnx\nprint(cytnx.__cpp_flags__)\")")

    




.. toctree::

