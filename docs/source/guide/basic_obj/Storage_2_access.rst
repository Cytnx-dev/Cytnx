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

.. code-block:: c++
    :linenos:

    auto A = cytnx::Storage(6);
    A.set_zeros();
    cout << A << endl;

    A.at<double>(4) = 4;
    cout << A << endl;


Output >>

.. code-block:: text

    dtype : Double (Float64)
    device: cytnx device: CPU
    size  : 6
    [ 0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 ]

    dtype : Double (Float64)
    device: cytnx device: CPU
    size  : 6
    [ 0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 4.00000e+00 0.00000e+00 ]


.. Note::
    
    1. The return value is the reference to the accessed element. This behavior is similar to a C++ *vector*. 
    2. The same template type as the dtype of the Storage instance should be specified when calling **at<>()**. If the types mismatch, an error will be prompted. 

* [New][v0.6.6+] The introduction of the Scalar class allows to get elements using **at()**  without type specialization (C++ only):

.. code-block:: c++
    :linenos:

    auto A = cytnx::Storage(6);
    cout << A << endl;

    Scalar elemt = A.at(4);
    cout << elemt << endl;

    A.at(4) = 4;
    cout << A << endl;

   
Output >>

.. code-block:: text

    dtype : Double (Float64)
    device: cytnx device: CPU
    size  : 6
    [ 0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 ]

    Scalar dtype: [Double (Float64)]
    0

    dtype : Double (Float64)
    device: cytnx device: CPU
    size  : 6
    [ 0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 4.00000e+00 0.00000e+00 ]




Get raw-pointer (C++ only)
***************************
In some cases the user might want to get the raw-pointer to the Storage. Cytnx provides two ways to get a raw-pointer. 

1. Use **Storage.data<>()**:
    If you use **.data<T>**, the template type *T* should match the dtype of the Storage. The return value will be a pointer with type *T*. 

* In C++:

.. code-block:: c++
    :linenos:

    auto A = cytnx::Storage(6);
    double *pA = A.data<double>();

    
2. Use **Storage.data()**:
    Using **.data()** without specialization returns a void pointer, please use with caution! 

* In C++:

.. code-block:: c++
    :linenos:

    auto A = cytnx::Storage(6);
    void *pA = A.data();


.. Note::

    If the current Storage instance is allocated on a GPU, the return pointer will be a device pointer. 
    See :cuda-mem:`the CUDA documentation <>`.

.. Warning::

    The return pointer is shared with the Storage instance. Thus its life time will be limited by that of the Storage instance. If the instance is destroyed first, the memory will be freed, and the pointer will be invalid as well. Please use with caution!  



.. toctree::
