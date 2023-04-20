Access elements
-----------------
To access the element in Storage, different API should be used in python and C++ due to the fundamental difference in two languages.


Get/Set element
****************
* In python, simply use **operator[]**:

.. code-block:: python 
    :linenos:

    A = cytnx.Storage(6)
    A.set_zeros()
    print(A)

    A[4] = 4
    print(A)



* In c++, we use **at<>()**:

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
    
    1. The return is the reference of the element, just like c++ *vector*. 
    2. The template type that match the dtype of Storage instance should be specify when calling **at<>()**. If the type mismatch, an error will be prompt. 

* [New][v0.6.6+] The introduce of C++ only Scalar class allow get elements using **at()**  without specialization (C++ only)

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
In some cases where user might want to get the raw-pointer from Storage. It is possible to do so. Cytnx provide two ways you can get a raw-pointer. 

1. Use **Storage.data<>()**:
    Using **.data<T>** should provide a template type *T* that match the dtype of Storage. The return will be a pointer with type *T*. 

* In c++:

.. code-block:: c++
    :linenos:

    auto A = cytnx::Storage(6);
    double *pA = A.data<double>();

    
2. Use **Storage.data()**:
    Using **.data()** without specialization return a void pointer, please use with caution. 

* In c++:

.. code-block:: c++
    :linenos:

    auto A = cytnx::Storage(6);
    void *pA = A.data();


.. Note::

    If the current Storage instance is allocate on GPU, the return pointer will be a device pointer. 
    See :cuda-mem:`this page <>`.

.. Warning::

    The return pointer is shared with the Storage instance. Thus it's life time will be the same as that instance. If the instance is destroy first, the memory will be free, and the pointer will be invalid as well. Please use with caution.  



.. toctree::
