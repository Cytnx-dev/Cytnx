Accessing elements
------------------
To access the elements in Storage, different APIs are used in Python and C++ due to the fundamental differences in the two languages.


Get/Set elements
****************
* In Python, simply use the **operator[]**:

.. code-block:: python 
    :linenos:

    A = cytnx.Storage(6)
    A.set_zeros()
    print(A)

    A[4] = 4
    print(A)



* In C++, use **at<>()**:

.. literalinclude:: ../../../code/cplusplus/guide_codes/4_2_1_ex1.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../../code/cplusplus/outputs/4_2_1_ex1.out
    :language: text

.. Note::
    
    1. The return value is the reference to the accessed element. This behavior is similar to a C++ *vector*. 
    2. The same template type as the dtype of the Storage instance should be specified when calling **at<>()**. If the types mismatch, an error will be prompted. 

* [New][v0.6.6+] The introduction of the Scalar class allows to get elements using **at()**  without type specialization (C++ only):

.. literalinclude:: ../../../code/cplusplus/guide_codes/4_2_1_ex2.cpp
    :language: c++
    :linenos:
   
Output >>

.. literalinclude:: ../../../code/cplusplus/outputs/4_2_1_ex2.out
    :language: text


Get raw-pointer (C++ only)
***************************
In some cases the user might want to get the raw-pointer to the Storage. Cytnx provides two ways to get a raw-pointer. 

1. Use **Storage.data<>()**:
    If you use **.data<T>**, the template type *T* should match the dtype of the Storage. The return value will be a pointer with type *T*. 

* In C++:

.. literalinclude:: ../../../code/cplusplus/guide_codes/4_2_2_ex1.cpp
    :language: c++
    :linenos:
    
2. Use **Storage.data()**:
    Using **.data()** without specialization returns a void pointer, please use with caution! 

* In C++:

.. literalinclude:: ../../../code/cplusplus/guide_codes/4_2_2_ex2.cpp
    :language: c++
    :linenos:

.. Note::

    If the current Storage instance is allocated on a GPU, the return pointer will be a device pointer. 
    See :cuda-mem:`the CUDA documentation <>`.

.. Warning::

    The return pointer is shared with the Storage instance. Thus its life time will be limited by that of the Storage instance. If the instance is destroyed first, the memory will be freed, and the pointer will be invalid as well. Please use with caution!  



.. toctree::
