2. Manipulate Tensor
----------------------
Next, let's look at the operations that are commonly used to manipulate Tensor object. 

2.1 reshape 
**********************
Suppose we want to create a rank-3 Tensor with shape=(2,3,4), starting with a rank-1 Tensor with shape=(24) initialized using **arange()**. 

This operation is called *reshape* 

We can use **Tensor.reshape** function to do this. 

* In python:

.. code-block:: python 
    :linenos:

    A = cytnx.arange(24)
    B = A.reshape(2,3,4)
    print(A)
    print(B)

* In C++:

.. code-block:: c++
    :linenos:

    auto A = cytnx::arange(24);
    auto B = A.reshape(2,3,4);
    cout << A << endl;
    cout << B << endl;
   
>> Output:

.. code-block:: text

    Total elem: 24
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (24)
    [0.00000e+00 1.00000e+00 2.00000e+00 3.00000e+00 4.00000e+00 5.00000e+00 6.00000e+00 7.00000e+00 8.00000e+00 9.00000e+00 1.00000e+01 1.10000e+01 1.20000e+01 1.30000e+01 1.40000e+01 1.50000e+01 1.60000e+01 1.70000e+01 1.80000e+01 1.90000e+01 2.00000e+01 2.10000e+01 2.20000e+01 2.30000e+01 ]

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
 

Notice that calling **reshape()** returns a new object *B*, so the original object *A* is not changed after calls reshape. 

There is the other function **Tensor.reshape_** (with a underscore) that also performs reshape, but instead of return a new reshaped object, it performs inplace reshape to the instance that calls the function. For example:

* In python:

.. code-block:: python
    :linenos:

    A = cytnx.arange(24)
    print(A)
    A.reshape_(2,3,4)
    print(A)

* In C++:

.. code-block:: c++
    :linenos:

    auto A = cytnx::arange(24);
    cout << A << endl;
    A.reshape_(2,3,4);
    cout << A << endl;

>> Output:

.. code-block:: text

    Total elem: 24
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (24)
    [0.00000e+00 1.00000e+00 2.00000e+00 3.00000e+00 4.00000e+00 5.00000e+00 6.00000e+00 7.00000e+00 8.00000e+00 9.00000e+00 1.00000e+01 1.10000e+01 1.20000e+01 1.30000e+01 1.40000e+01 1.50000e+01 1.60000e+01 1.70000e+01 1.80000e+01 1.90000e+01 2.00000e+01 2.10000e+01 2.20000e+01 2.30000e+01 ]

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

Thus we see that using underscore version modify the instance itself. 


.. Note::

    In general, all the funcions in Cytnx that end with a underscore _ is either a inplace function that modify the instance that calls it, or return the reference of some class member. 

.. Hint::

    You can use **Tensor.shape()** to get the shape of Tensor.

2.1 permute
**********************
Now, let's again use the same rank-3  with shape=(2,3,4) as example. This time we want to do permute on the Tensor to exchange axes from indices (0,1,2)->(1,2,0)

This can be achieved with **Tensor.permute** 

* In python:

.. code-block:: python 
    :linenos:
    
    A = cytnx.arange(24).reshape(2,3,4)
    B = A.permute(1,2,0)
    print(A)
    print(B)

* In c++:

.. code-block:: c++ 
    :linenos:

    auto A = cytnx::arange(24).reshape(2,3,4);
    auto B = A.permute(1,2,0);
    cout << A << endl;
    cout << B << endl;

>> Output:

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

    Total elem: 24
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (3,4,2)
    [[[0.00000e+00 1.20000e+01 ]
      [1.00000e+00 1.30000e+01 ]
      [2.00000e+00 1.40000e+01 ]
      [3.00000e+00 1.50000e+01 ]]
     [[4.00000e+00 1.60000e+01 ]
      [5.00000e+00 1.70000e+01 ]
      [6.00000e+00 1.80000e+01 ]
      [7.00000e+00 1.90000e+01 ]]
     [[8.00000e+00 2.00000e+01 ]
      [9.00000e+00 2.10000e+01 ]
      [1.00000e+01 2.20000e+01 ]
      [1.10000e+01 2.30000e+01 ]]]

.. Note::

    Just like before, there is an equivalent **Tensor.permute_** end with underscore that performs inplace permute on the instance that calls it. 


.. Hint::
    
    In some situation where we don't want to create a copy of object, using inplace version of functions can reduce the memory usage.


In Cytnx, the permute operation does not moving the elements in the memory immediately. Only the meta-data that is seen by user are changed. 
This can avoid the redudant moving of elements. Note that this approach is also taken in `numpy.array <https://numpy.org/doc/1.18/reference/generated/numpy.array.html>`_ and `torch.tensor <https://pytorch.org/docs/stable/tensors.html>`_ .

If the meta-data is distached from the real memery layout, we call the Tensor in this status *non-contiguous*. We can use **Tensor.is_contiguous()** to check if the current Tensor is in contiguous status. 

You can force the Tensor to return to it's contiguous status by calling **Tensor.contiguous()/Tensor.contiguous_()**, although generally you don't have to worry about contiguous, as cytnx automatically handles it for you. 


* In python:

.. code-block:: python 
    :linenos:

    A = cytnx.arange(24).reshape(2,3,4)
    print(A.is_contiguous())
    print(A) 

    A.permute_(1,0,2)
    print(A.is_contiguous())
    print(A) 

    A.contiguous_()
    print(A.is_contiguous())

* In C++:

.. code-block:: c++
    :linenos:

    auto A = cytnx::arange(24).reshape(2,3,4);
    cout << A.is_contiguous() << endl;
    cout << A << endl;

    A.permute_(1,0,2);
    cout << A.is_contiguous() << endl;
    cout << A << endl;

    A.contiguous_();
    cout << A.is_contiguous() << endl;

Output>> 

.. code-block:: text

    True

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

    False

    Total elem: 24
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (3,2,4)
    [[[0.00000e+00 1.00000e+00 2.00000e+00 3.00000e+00 ]
      [1.20000e+01 1.30000e+01 1.40000e+01 1.50000e+01 ]]
     [[4.00000e+00 5.00000e+00 6.00000e+00 7.00000e+00 ]
      [1.60000e+01 1.70000e+01 1.80000e+01 1.90000e+01 ]]
     [[8.00000e+00 9.00000e+00 1.00000e+01 1.10000e+01 ]
      [2.00000e+01 2.10000e+01 2.20000e+01 2.30000e+01 ]]]

    True


.. Tip::

    1. Generally, you don't have to worry about contiguous issue. you can access the elements and call linalg just like this contiguous/non-contiguous thing doesn't exist. 
    
    2. In the case where the function does require user to manually make the Tensor contiguous, a warning will be prompt, and you can simply add a **Tensor.contiguous()/.contiguous_()** before the function call. 

    
.. Note::
    
    As metioned before, **Tensor.contiguous_()** (with underscore) make the current instance contiguous, while **Tensor.contiguous()** return a new object with contiguous status. 
    In the case where the current instance is already in it's contiguous status, calling contiguous will return itself, and no new object will be created. 




.. toctree::
