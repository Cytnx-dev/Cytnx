Install Cytnx
------------------
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

    $conda create --name cytnx python=3.7


.. note:: 

    * See :virtualenv:`This page <>` for how to use virtual enviroment in conda. 
    * Use can select the python version you want. we recommend using >=3.7

3. activate enviroment and conda install cytnx:
    
    Once you create a virtual enviroment, we need to activate enviroment before start using it. 

.. code-block:: shell

    $conda activate cytnx
    
.. code-block:: shell
    
    $conda install -c kaihsinwu cytnx_37

.. note::

    * depending on your version of python, you need to replace the cytnx_* to the one corresponds to your python version. 

    For example:
        * python=3.6 -> cytnx_36
        * python=3.7 -> cytnx_37
        * pyhton=3.8 -> cytnx_38

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

* The first one **cytnx.__cpp_include__** gives you the cytnx header files directory path.
* The second one **cytnx.__cpp_lib__** gives you the cytnx library file directory path. 
* The thrid one **cytnx.__cpp_linkflags__** gives you the essential linking flags that are required when you link your own programs that using cytnx. 


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

    g++ -std=c++11 -I${CYTNX_INC} test.cpp ${CYTNX_LIB}/libcytnx.a ${CYTNX_LINK} -o test


The first three lines are the python inline execution to get the three attributes and store them into **CYTNX_INC**, **CYTNX_LIB** and **CYTNX_LINK** variables. The last line is the standard simple compiling of the C++ code **test.cpp**. After execute these steps, we can then run this program with the executable **test**. 


.. code-block:: shell
    
    ./test


* Output:

.. code-block:: text
    
    Total elem: 4
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (4)
    [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]
    

%Build from source
%*********************
%For advanced user who wish to build cytnx from source, we provides the cmake install. 


.. toctree::

