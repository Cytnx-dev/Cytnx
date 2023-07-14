Accessing elements
----------------------
Next, let's take a look on how we can access elements of a Tensor.

Get elements 
***************************
On the Python side, we can simply use *slice* to get the elements, just as common with list/numpy.array/torch.tensor in Python. See :numpy-slice:`This page <>` for more details.
In C++, Cytnx ports this approach from Python to the C++ API. You can simply use a **slice string** to access elements. 

For example:

* In Python:

.. code-block:: python
    :linenos:

    A = cytnx.arange(24).reshape(2,3,4)
    print(A)

    B = A[0,:,1:4:2]
    print(B)

    C = A[:,1]    
    print(C)

* In C++:

.. literalinclude:: ../../../code/cplusplus/guide_codes/3_3_1_ex1.cpp
    :language: c++
    :linenos:

Output>>

.. literalinclude:: ../../../code/cplusplus/outputs/3_3_1_ex1.out
    :language: text

.. Note::

    1. To convert between Python and C++ APIs, notice that in C++ you need to use operator() instead of operator[] if you are using slice strings to access elements. 
    2. The return value will always be a Tensor object, even it only contains one element.


In the case where you have only one element in a Tensor, you can use **item()** to get the element as a standard Python/C++ type. 

* In Python:

.. code-block:: python
    :linenos:
    
    A = cytnx.arange(24).reshape(2,3,4)
    B = A[0,0,1]
    C = B.item()
    print(B)
    print(C)

   
* In C++:

.. literalinclude:: ../../../code/cplusplus/guide_codes/3_3_1_ex2.cpp
    :language: c++
    :linenos:

Output>> 

.. literalinclude:: ../../../code/cplusplus/outputs/3_3_1_ex2.out
    :language: text

.. Note::
    
    1. In C++, using **item<>()** to get the element requires to explicitly specify the type that matches the dtype of the Tensor. If the type specifier does not match, an error will be prompted. 
    2. Starting from v0.7+, users can use item() in C++ without explicitly specifying the type with a template. 


Set elements
***************************
Setting elements is pretty much the same as in numpy.array/torch.tensor. You can assign a Tensor to a specific slice, or set all the elements in that slice to be the same value. 

For example:

* In Python:

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
    
* In C++:

.. literalinclude:: ../../../code/cplusplus/guide_codes/3_3_2_ex1.cpp
    :language: c++
    :linenos:

Output>>

.. literalinclude:: ../../../code/cplusplus/outputs/3_3_2_ex1.out
    :language: text



Low-level API (C++ only) 
*******************************
On the C++ side, Cytnx provides lower-level APIs with slightly smaller overhead for getting elements. 
These low-level APIs require using an **Accessor** object. 

* Accessor:
    **Accessor** object is equivalent to Python *slice*. It is sometimes convenient to use aliases to simplify the expression when using it.
    
    .. code-block:: C++
        :linenos:

            typedef ac=cytnx::Accessor;

            ac(4);     // this is equal to index '4' in Python
            ac::all(); // this is equal to ':' in Python 
            ac::range(0,4,2); // this is equal to '0:4:2' in Python 



In the following, let's see how it can be used to get/set the elements from/in a Tensor.

1. operator[] (middle level API) :
    
.. code-block:: c++
    :linenos:

        typedef ac=cytnx::Accessor;
        auto A = cytnx::arange(24).reshape(2,3,4);
        auto B = cytnx::zeros({3,2});

        // [get] this is equal to A[0,:,1:4:2] in Python:
        auto C = A[{ac(0},ac::all(),ac::range(1,4,2)}];
        
        // [set] this is equal to A[1,:,0:4:2] = B in Python:
        A[{ac(1),ac::all(),ac::range(0,4,2)}] = B;


.. Note::

    Remember to write a braket{} around the elements to be accessed. This is needed because the C++ operator[] can only accept one argument. 


2. get/set (low level API) :
    get() and set() are part of the low-level API. Operator() and Operator[] are all built based on these.
    
.. code-block:: c++
    :linenos:

        typedef ac=cytnx::Accessor;
        auto A = cytnx::arange(24).reshape(2,3,4);
        auto B = cytnx::zeros({3,2});

        // [get] this is equal to A[0,:,1:4:2] in Python:
        auto C = A.get({ac(0},ac::all(),ac::range(1,4,2)});
        
        // [set] this is equal to A[1,:,0:4:2] = B in Python:
        A.set({ac(1),ac::all(),ac::range(0,4,2)}, B);



.. Hint::

    1. Similarly, you can also pass a C++ *vector<cytnx_int64>* as an argument. 

.. Tip::

    If your code makes frequent use of get/set elements, using the low-level API can reduce the overhead.



.. toctree::
