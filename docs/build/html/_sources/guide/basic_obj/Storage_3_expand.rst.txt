Increase size
-----------------
Just like C++ vectors, we can increase the size of the Storage.

Append
********
It is possible to append a new element to the end of the Storage. 
For example

* In Python:

.. code-block:: python 
    :linenos:
    
    A = cytnx.Storage(4)
    A.set_zeros();
    print(A)

    A.append(500)
    print(A)    
    
   
* In C++:

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
Equivalently to *vector.resize* in C++, we can resize the Storage in Cytnx.

* In Python:

.. code-block:: python
    :linenos:

    A = cytnx.Storage(4);
    print(A.size());

    A.resize(5);
    print(A.size());

* In C++:

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
    
    [Deprecated] If the size is increased in the resize operation, the additional elements will NOT be set to zero. Please use with care. 

    [New][v0.6.6+] Additional elements are initialized by zeros when the memory is increased by resize. This behavior is similar to that of a vector.

.. Tip::

    1. You can use **Storage.size()** to get the current number of elements in the Storage.
    2. Internally, Cytnx allocates memory in multiples of 2. This optimizes the bandwidth use of CPU/GPU transfers and possibly increases the performance of some kernels. You can use **Storage.capacity()** to check the currently allocated number of elements in real memory, which might be larger than the number of elements in the Storage. 



.. toctree::
