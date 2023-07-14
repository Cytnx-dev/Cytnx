From/To C++.vector
--------------------
Cytnx provides a way to convert directly between a C++ *vector* and a Storage instance. 


To convert a C++ vector to a Storage, use **Storage::from_vector**:

* In C++

.. literalinclude:: ../../../code/cplusplus/guide_codes/4_4_ex1.cpp
    :language: c++
    :linenos:

Output >>

.. code-block:: text

    dtype : Double (Float64)
    device: cytnx device: CPU
    size  : 4
    [ 6.00000e+00 6.00000e+00 6.00000e+00 6.00000e+00 ]

    dtype : Double (Float64)
    device: cytnx device: CUDA/GPU-id:0
    size  : 4
    [ 6.00000e+00 6.00000e+00 6.00000e+00 6.00000e+00 ]

.. Note::

    You can also specify the device upon calling *from_vector*. 

.. Tip::

    Cytnx overloads the **operator<<** for C++ vectors. You can directly print any vector when **using namespace cytnx;**.  
    Alternatively, you can also use the **print()** function just like in Python.



[New][v0.7.5+]
To convert a Storage to std::vector with type *T*, use **Storage.vector<T>()**:


* In C++

.. literalinclude:: ../../../code/cplusplus/guide_codes/4_4_ex2.cpp
    :language: c++
    :linenos:
    
Output >>

.. literalinclude:: ../../../code/cplusplus/outputs/4_4_ex2.out
    :language: text

.. Note::

    The type T has to match the dtype of the Storage, otherwise an error will be raised. 



.. toctree::
