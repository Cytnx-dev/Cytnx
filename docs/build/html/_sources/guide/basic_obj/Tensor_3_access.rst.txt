Access elements
----------------------
Next, let's take a look on how we can access elements inside a Tensor.

Get elements 
***************************
Just like python list/numpy.array/torch.tensor, on the python side, we can simply use *slice* to get the elements. See :numpy-slice:`This page <>` .
In c++, cytnx take this approach from python and bring it to our C++ API. You can simply use the **slice string** to access elements. 

For example:

* In python:

.. code-block:: python
    :linenos:

    A = cytnx.arange(24).reshape(2,3,4)
    print(A)

    B = A[0,:,1:4:2]
    print(B)

    C = A[:,1]    
    print(C)

* In C++:

.. code-block:: c++
    :linenos:

    auto A = cytnx::arange(24).reshape(2,3,4);
    cout << A << endl;

    auto B = A(0,":","1:4:2");
    cout << B << endl;

    auto C = A(":",1);    
    cout << C << endl;

Output>>

.. code-block:: text

    Total elem: 24
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (2,3,4)
    [[[0.00000e+00 1.00000e+00 2.00000e+00 3.00000e+00 ]
      [4.00000e+00 5.00000e+00 6.00000e+00 7.00000e+00 ]
      [8.00000e+00 9.00000e+00 1.00000e+01 1.10000e+01 ]]
     [[1.20000e+01 1.30000e+01 1.40000e+01 1.50000e+01 ]
      [1.60000e+01 1.70000e+01 1.80000e+01 1.90000e+01 ]
      [2.00000e+01 2.10000e+01 2.20000e+01 2.30000e+01 ]]]


    Total elem: 6
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (3,2)
    [[1.00000e+00 3.00000e+00 ]
     [5.00000e+00 7.00000e+00 ]
     [9.00000e+00 1.10000e+01 ]]


    Total elem: 8
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (2,4)
    [[4.00000e+00 5.00000e+00 6.00000e+00 7.00000e+00 ]
     [1.60000e+01 1.70000e+01 1.80000e+01 1.90000e+01 ]]
    

.. Note::

    1. To convert in between python and C++ APIs, notice that in C++, we use operator() instead of operator[] if you are using slice string to acess elements. 
    2. The return will always be Tensor object, even it is only one elements in the Tensor.


In the case where you have only one element in a Tensor, we can use **item()** to get the element in the standard python type/c++ type. 

* In python:

.. code-block:: python
    :linenos:
    
    A = cytnx.arange(24).reshape(2,3,4)
    B = A[0,0,1]
    C = B.item()
    print(B)
    print(C)

   
* In C++:

.. code-block:: c++
    :linenos:

    auto A = cytnx::arange(24).reshape(2,3,4);
    auto B = A(0,0,1);
    Scalar C = B.item(); 
    double Ct = B.item<double>();

    cout << B << endl;
    cout << C << endl;
    cout << Ct << endl;

Output>> 

.. code-block:: text 

    Total elem: 1
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (1)
    [1.00000e+00 ]


    1.0

.. Note::
    
    1. In C++, using **item<>()** to get the element require explicitly specify the type that match the dtype of the Tensor. If the type specify does not match, an error will be prompt. 
    2. Starting from v0.7+, user can use item() in C++ without explificly specify type with template. 


Set elememts
***************************
Setting elements is pretty much the same as numpy.array/torch.tensor. You can assign a Tensor to a specific slice, our set all the elements in that slice to be the same value. 

For example:

* In python:

.. code-block:: python
    :linenos:

    A = cytnx.arange(24).reshape(2,3,4)
    B = cytnx.zeros([3,2])
    print(A)
    print(B)

    A[1,:,::2] = B
    print(A)

    A[0,::2,2] = 4
    print(A)
    
* In c++:

.. code-block:: c++
    :linenos:

    auto A = cytnx::arange(24).reshape(2,3,4);
    auto B = cytnx::zeros({3,2});
    cout << A << endl;
    cout << B << endl;

    A(1,":","::2") = B;
    cout << A << endl;

    A(0,"::2",2) = 4;
    cout << A << endl;

Output>>

.. code-block:: text

    Total elem: 24
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (2,3,4)
    [[[0.00000e+00 1.00000e+00 2.00000e+00 3.00000e+00 ]
      [4.00000e+00 5.00000e+00 6.00000e+00 7.00000e+00 ]
      [8.00000e+00 9.00000e+00 1.00000e+01 1.10000e+01 ]]
     [[1.20000e+01 1.30000e+01 1.40000e+01 1.50000e+01 ]
      [1.60000e+01 1.70000e+01 1.80000e+01 1.90000e+01 ]
      [2.00000e+01 2.10000e+01 2.20000e+01 2.30000e+01 ]]]


    Total elem: 6
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (3,2)
    [[0.00000e+00 0.00000e+00 ]
     [0.00000e+00 0.00000e+00 ]
     [0.00000e+00 0.00000e+00 ]]


    Total elem: 24
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (2,3,4)
    [[[0.00000e+00 1.00000e+00 2.00000e+00 3.00000e+00 ]
      [4.00000e+00 5.00000e+00 6.00000e+00 7.00000e+00 ]
      [8.00000e+00 9.00000e+00 1.00000e+01 1.10000e+01 ]]
     [[0.00000e+00 1.30000e+01 0.00000e+00 1.50000e+01 ]
      [0.00000e+00 1.70000e+01 0.00000e+00 1.90000e+01 ]
      [0.00000e+00 2.10000e+01 0.00000e+00 2.30000e+01 ]]]


    Total elem: 24
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (2,3,4)
    [[[0.00000e+00 1.00000e+00 4.00000e+00 3.00000e+00 ]
      [4.00000e+00 5.00000e+00 6.00000e+00 7.00000e+00 ]
      [8.00000e+00 9.00000e+00 4.00000e+00 1.10000e+01 ]]
     [[0.00000e+00 1.30000e+01 0.00000e+00 1.50000e+01 ]
      [0.00000e+00 1.70000e+01 0.00000e+00 1.90000e+01 ]
      [0.00000e+00 2.10000e+01 0.00000e+00 2.30000e+01 ]]]




Low-level API (C++ only) 
*******************************
On C++ side, cytnx provide lower-level APIs with slightly smaller overhead for getting elements. 
These low-level APIs require using with **Accessor** object. 

* Accessor:
    **Accessor** object is equivalent to python *slice*. It is sometimes convenient to use alias to simplify the expression when using it.
    
    .. code-block:: C++
        :linenos:

            typedef ac=cytnx::Accessor;

            ac(4);     // this equal to index '4' in python
            ac::all(); // this equal to ':' in python 
            ac::range(0,4,2); // this equal to '0:4:2' in python 



In the following, let's see how it can be used to get/set the elements from/to Tensor.

1. operator[] (middle level API) :
    
.. code-block:: c++
    :linenos:

        typedef ac=cytnx::Accessor;
        auto A = cytnx::arange(24).reshape(2,3,4);
        auto B = cytnx::zeros({3,2});

        // [get] this is equal to A[0,:,1:4:2] in python:
        auto C = A[{ac(0},ac::all(),ac::range(1,4,2)}];
        
        // [set] this is equal to A[1,:,0:4:2] = B in python:
        A[{ac(1),ac::all(),ac::range(0,4,2)}] = B;


.. Note::

    Remember to put a braket{}. This because C++ operator[] can only accept one argument. 


2. get/set (lowest level API) :
    get() and set() is the lowest-level API. Operator() and Operator[] are all build base on these.
    
.. code-block:: c++
    :linenos:

        typedef ac=cytnx::Accessor;
        auto A = cytnx::arange(24).reshape(2,3,4);
        auto B = cytnx::zeros({3,2});

        // [get] this is equal to A[0,:,1:4:2] in python:
        auto C = A.get({ac(0},ac::all(),ac::range(1,4,2)});
        
        // [set] this is equal to A[1,:,0:4:2] = B in python:
        A.set({ac(1),ac::all(),ac::range(0,4,2)}, B);



.. Hint::

    1. Similarly, you can also pass a c++ *vector<cytnx_int64>* as argument. 

.. Tip::

    If your code requires frequently get/set elements, using low-level API can reduce the overhead.



.. toctree::
