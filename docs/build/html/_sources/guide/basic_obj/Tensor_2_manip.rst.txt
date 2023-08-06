Manipulating Tensors
----------------------
Next, let's look at the operations that are commonly used to manipulate Tensor objects. 

reshape 
**********************
Suppose we want to create a rank-3 Tensor with shape=(2,3,4), starting with a rank-1 Tensor with shape=(24) initialized using **arange()**. 

This operation is called *reshape* 

We can use the **Tensor.reshape** function to do this. 

* In Python:

.. code-block:: python 
    :linenos:

    A = cytnx.arange(24)
    B = A.reshape(2,3,4)
    print(A)
    print(B)

* In C++:

.. literalinclude:: ../../../code/cplusplus/guide_codes/3_2_1_ex1.cpp
    :language: c++
    :linenos:
   
>> Output:

.. literalinclude:: ../../../code/cplusplus/outputs/3_2_1_ex1.out
    :language: text

Notice that calling **reshape()** returns a new object *B*, so the original object *A*'s shape is not changed after calling reshape. 

The function **Tensor.reshape_** (with a underscore) performs a reshape as well, but instead of returning a new reshaped object, it performs an inplace reshape of the instance that calls the function. For example:

* In Python:

.. code-block:: python
    :linenos:

    A = cytnx.arange(24)
    print(A)
    A.reshape_(2,3,4)
    print(A)

* In C++:
.. literalinclude:: ../../../code/cplusplus/guide_codes/3_2_1_ex2.cpp
    :language: c++
    :linenos:

>> Output:

.. literalinclude:: ../../../code/cplusplus/outputs/3_2_1_ex2.out
    :language: text

Thus, we see that using the underscore version modifies the original Tensor itself. 


.. Note::

    In general, all the funcions in Cytnx that end with an underscore _ are either inplace functions that modify the instance that calls it, or return a reference of some class member. 

.. Hint::

    You can use **Tensor.shape()** to get the shape of a Tensor.

permute
**********************
Let's consider the same rank-3 Tensor with shape=(2,3,4) as an example. This time we want to permute the order of the Tensor indices according to (0,1,2)->(1,2,0)

This can be achieved with **Tensor.permute** 

* In Python:

.. code-block:: python 
    :linenos:
    
    A = cytnx.arange(24).reshape(2,3,4)
    B = A.permute(1,2,0)
    print(A)
    print(B)

* In C++:

.. literalinclude:: ../../../code/cplusplus/guide_codes/3_2_2_ex1.cpp
    :language: c++
    :linenos:

>> Output:

.. literalinclude:: ../../../code/cplusplus/outputs/3_2_2_ex1.out
    :language: text

.. Note::

    Just like before, there is an equivalent **Tensor.permute_**, which ends with an underscore, that performs an inplace permute on the instance that calls it. 


In Cytnx, the permute operation does not move the elements in the memory immediately. Only the meta-data that is seen by the user is changed. 
This can avoid the redundant moving of elements. Note that this approach is also used in :numpy-arr:`numpy.array <>` and :torch-tn:`torch.tensor <>` .

After the permute, the meta-data does not correspond to the memory order anymore. If the meta-data is distached that way from the real memory layout, we call the Tensor in this status *non-contiguous*. We can use **Tensor.is_contiguous()** to check if the current Tensor is in contiguous status. 

You can force the Tensor to become contiguous by calling **Tensor.contiguous()** or **Tensor.contiguous_()**. The memory is then rearranged according to the shape of the Tensor. Generally you do not have to worry about the contiguous status, as Cytnx automatically handles it for you.


* In Python:

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

.. literalinclude:: ../../../code/cplusplus/guide_codes/3_2_2_ex2.cpp
    :language: c++
    :linenos:

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

    1. Generally, you don't have to worry about contiguous issues. You can access the elements and call linalg just like this contiguous/non-contiguous property does not exist. 
    
    2. In cases where a function does require the user to manually force the Tensor to be contiguous, a warning will be prompted, and you can simply add a **Tensor.contiguous()** or **Tensor.contiguous_()** before the function call.

    3. Making a Tensor contiguous involves copying the elements in memory and can slow down the algorithm. Unnecessary calls of **Tensor.contiguous()** or **Tensor.contiguous_()** should therefore be avoided.
    
    4. See :ref:`Contiguous` for more details about the contiguous status.

    
.. Note::
    
    As mentioned before, **Tensor.contiguous_()** (with underscore) makes the current instance contiguous, while **Tensor.contiguous()** returns a new object with contiguous status. 
    In the case that the current instance is already in it's contiguous status, calling contiguous will return itself, and no new object will be created. 









.. toctree::
