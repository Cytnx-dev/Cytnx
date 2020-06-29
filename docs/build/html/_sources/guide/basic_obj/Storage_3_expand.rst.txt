Increase size
-----------------
Just like C++ vector, we can increase the size of Storage.

Append
********
This append a new element at the end of the Storage. 
For example

* In python:

.. code-block:: python 
    :linenos:
    
    A = cytnx.Storage(4)
    A.set_zeros();
    print(A)

    A.append(500)
    print(A)    
    
   
* In c++:

.. code-block:: c++
    :linenos:

    auto A = cytnx::Storage(4);
    A.set_zeros();
    cout << A << endl;

    A.append(500);
    cout << A << endl;

Output>>

.. code-block:: text

    dtype : Double (Float64)
    device: cytnx device: CPU
    size  : 4
    [ 0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 ]

    dtype : Double (Float64)
    device: cytnx device: CPU
    size  : 5
    [ 0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 5.00000e+02 ]


Resize
********
Equvalent to c++ *vector.resize*, we can do the same thing in cytnx.

* In python:

.. code-block:: python
    :linenos:

    A = cytnx.Storage(4);
    print(A.size());

    A.resize(5);
    print(A.size());

* In c++:

.. code-block:: c++
    :linenos:

    auto A = cytnx::Storage(4);
    cout << A.size() << endl;

    A.resize(5);    
    cout << A.size() << endl;

Output>>

.. code-block:: text

    4
    5

.. Note::
    
    If the size is increase after resize, the additional elements will NOT be set to zero. Please be careful. 

.. Tip::

    1. You can use **Storage.size()** to get the current size of Storage.
    2. Internally, cytnx allocate memory in multiple of 32. This choice is to optimize the bandwidth of CPU/GPU transfer and possibly performance of some kernels. you can use **Storage.capacity()** to check the current real memory size. 



.. toctree::
