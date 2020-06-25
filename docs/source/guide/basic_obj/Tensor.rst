Tensor
==========
Tensor is the basic building block of Cytnx. 
In fact, the API of Tensor in cytnx is very similar to `torch.tensor <https://pytorch.org/docs/stable/tensors.html>`_ (so numpy.array, since they are also similar to each other)


Let's take a look on how to use it.


1. Define a Tensor
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

    1. In cytnx, the conversion of python list will equivalent to C++ *vector* or in some case like here, it is a *initializer list*. 

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

In python:

.. code-block:: python

    A = cytnx.zeros([3,4,5],dtype=cytnx.Type.Int64,device=cytnx.Device.cuda)

In c++:

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



2. Manipulate Tensor
----------------------
Next, let's look at the operations that are commonly used to manipulate Tensor object. 

2.1 permute & reshape
**********************
Consider a rank-4 Tensor with shape (3,4,5,6) as example. 


.. toctree::



