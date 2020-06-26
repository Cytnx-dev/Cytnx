Tensor
==========
Tensor is the basic building block of Cytnx. 
In fact, the API of Tensor in cytnx is very similar to `torch.tensor <https://pytorch.org/docs/stable/tensors.html>`_ (so as numpy.array, since they are also similar to each other)


Let's take a look on how to use it.


1. Create a Tensor
-------------------
Just like `numpy.array <https://numpy.org/doc/1.18/reference/generated/numpy.array.html>`_ / `torch.tensor <https://pytorch.org/docs/stable/tensors.html>`_, Tensor is generally created using generator such as **zero()**, **arange()**, **ones()**.

For example, suppose we want to define a rank-3 tensor with shape (3,4,5), and initialize all elements with zero:

* In python:

.. code-block:: python
    :linenos:

    A = cytnx.zeros([3,4,5]);
        

* In c++:

.. code-block:: c++
    :linenos:

    cytnx::Tensor A = cytnx::zeros({3,4,5});

.. Note::

    1. In cytnx, the conversion of python list is equivalent to C++ *vector*; or in some case like here, it is a *initializer list*. 

    2. The conversion in between is pretty straight forward, one simply replace [] in python with {}, and you are all set!



Other options such as **arange()** (similar as np.arange), and **ones** (similar as np.ones) can also be done. 

* In python : 

.. code-block:: python 
    :linenos:

    A = cytnx.arange(10);     #rank-1 Tensor from [0,10) with step 1
    B = cytnx.arange(0,10,2); #rank-1 Tensor from [0,10) with step 2
    C = cytnx.ones([3,4,5]);  #Tensor of shape (3,4,5) with all elements set to one.

* In c++:

.. code-block:: c++
    :linenos:

    auto A = cytnx::arange(10);     //rank-1 Tensor from [0,10) with step 1
    auto B = cytnx::arange(0,10,2); //rank-1 Tensor from [0,10) with step 2
    auto C = cytnx::ones({3,4,5});  //Tensor of shape (3,4,5) with all elements set to one.


:Tips: In C++, you could make use of *auto* to simplify your code! 


1.1. Tensor with different dtype and device 
*******************************************
By default, the Tensor will be created with *double* type (or *float* in python) on CPU if there is no additional arguments provided upon creating the Tensor. 

You can create a Tensor with different data type, and/or on different devices simply by specify the **dtype** and the **device** arguments upon initialization. For example, the following codes create a Tensor with 64bit integer on cuda-enabled GPU. 

* In python:

.. code-block:: python

    A = cytnx.zeros([3,4,5],dtype=cytnx.Type.Int64,device=cytnx.Device.cuda)

* In c++:

.. code-block:: c++

    auto A = cytnx.zeros({3,4,5},cytnx::Type::Int64,cytnx::Device::cuda);

.. Note:: 

    1. Remember the difference of . in python and :: in C++ when you use Type and Device classes. 
    2. If you have multiple GPUs, you can specify which GPU you want to init Tensor by adding gpu-id to cytnx::Device::cuda. 
        
        For example: 
        
            device=cytnx.Device.cuda+2   #will create Tensor on GPU id=2

            device=cytnx.Device.cuda+4   #will create Tensor on GPU id=4

    3. In C++, there is no keyword argument as python, so make sure you put the argument in the correct order. Check `API documentation <https://kaihsin.github.io/Cytnx/docs/html/index.html>`_ for function signatures!  


Currently, there are several data types supported by cytnx:

.. tabularcolumns:: |l|l|l|

+------------------+----------------------+-------------------+
| cytnx type       | c++ type             | Type object       |
+==================+======================+===================+
| cytnx_double     | double               | Type.Double       |
+------------------+----------------------+-------------------+
| cytnx_float      | float                | Type.Float        |
+------------------+----------------------+-------------------+
| cytnx_uint64     | uint64_t             | Type.Uint64       |
+------------------+----------------------+-------------------+
| cytnx_uint32     | uint32_t             | Type.Uint32       |
+------------------+----------------------+-------------------+
| cytnx_uint16     | uint16_t             | Type.Uint16       |
+------------------+----------------------+-------------------+
| cytnx_int64      | int64_t              | Type.Int64        |
+------------------+----------------------+-------------------+
| cytnx_int32      | int32_t              | Type.Int32        |
+------------------+----------------------+-------------------+
| cytnx_int16      | int16_t              | Type.Int16        |
+------------------+----------------------+-------------------+
| cytnx_complex128 | std::complex<double> | Type.ComplexDouble|
+------------------+----------------------+-------------------+
| cytnx_complex64  | std::complex<float>  | Type.ComplexFloat |
+------------------+----------------------+-------------------+
| cytnx_bool       | bool                 | Type.Bool         |
+------------------+----------------------+-------------------+


For devices, Cytnx currently supports

.. tabularcolumns:: |l|l|

+------------------+----------------------+
| cytnx type       | Device object        |
+==================+======================+
| CPU              | Device.cpu           | 
+------------------+----------------------+
| CUDA-enabled GPU | Device.cuda+x        |
+------------------+----------------------+

1.2 Type conversion 
**********************
It is possible to convert a Tensor to a different data type. To convert the data type, simply use **Tensor.astype()**.

For example, consider a Tensor *A* with **dtype=Type.Int64**, and we want to convert it to **Type.Double**

* In python:

.. code-block:: python 
    :linenos:
    
    A = cytnx.ones([3,4],dtype=cytnx.Type.Int64)
    B = A.astype(cytnx.Type.Double)
    print(A.dtype_str())
    print(B.dtype_str())

* In c++:

.. code-block:: c++
    :linenos:

    auto A = cytnx::ones({3,4},cytnx::Type::Int64);
    auto B = A.astype(cytnx::Type::Double);
    cout << A.dtype_str() << endl;
    cout << B.dtype_str() << endl;

>> Output:

.. code-block:: text
    
    Int64
    Double (Float64)



.. Note::
    
    1. Use Tensor.dtype() will return a type-id, where Tensor.dtype_str() will return the type name. 
    2. Complex data type cannot directly convert to real data type. Use Tensor.real()/Tensor.imag() if you want to get the real/imag part.


1.3 Transfer btwn devices
***************************
To move a Tensor between different devices is very easy. We can use **Tensor.to()** to move the Tensor to a different device.

For example, let's create a Tensor on cpu and transfer to GPU with gpu-id=0. 

* In python:

.. code-block:: python 
    :linenos:

    A = cytnx.ones([2,2]) #on CPU
    print(A)
    A.to(cytnx.Device.cuda+0)
    print(A)

* In c++:

.. code-block:: c++
    :linenos:

    auto A = cytnx::ones([2,2]); //on CPU
    cout << A << endl;
    A.to(cytnx.Device.cuda+0);
    cout << A << endl;

>> Output:

.. code-block:: text

    Total elem: 4
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (2,2)
    [[1.00000e+00 1.00000e+00 ]
     [1.00000e+00 1.00000e+00 ]]

    Total elem: 4
    type  : Double (Float64)
    cytnx device: CUDA/GPU-id:0
    Shape : (2,2)
    [[1.00000e+00 1.00000e+00 ]
     [1.00000e+00 1.00000e+00 ]]


.. Note::
    
    You can use **Tensor.device()** to get the current device-id (cpu = -1), where as **Tensor.device_str()** returns the device name. 


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





.. toctree::



