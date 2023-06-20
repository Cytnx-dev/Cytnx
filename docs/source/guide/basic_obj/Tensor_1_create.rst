Creating a Tensor
-------------------
When creating a Tensor, its elements are initialized with either fixed values, random elements, or from existing data.

Initialized Tensor
************************
Just like with :numpy-arr:`numpy.array <>` / :torch-tn:`torch.tensor <>`, a Tensor is generally created using a generator such as **zero()**, **arange()**, **ones()** or **eye()**.

For example, suppose we want to define a rank-3 tensor with shape (3,4,5), and initialize all elements with zero:

* In Python:

.. code-block:: python
    :linenos:

    A = cytnx.zeros([3,4,5]);
        

* In C++:

.. code-block:: c++
    :linenos:

    cytnx::Tensor A = cytnx::zeros({3,4,5});

.. Note::

    1. In Cytnx, a Python list is equivalent to a C++ *vector*; or in some cases like here, it is an *initializer list*. 

    2. The conversion between Python and C++ is pretty straight forward, one simply replaces [] in Python with {}, and you are all set!



Tensors can also be created and initialized with **arange()** (similar as np.arange), **ones** (similar as np.ones) or **eye** (identity matrix):

* In Python : 

.. code-block:: python 
    :linenos:

    A = cytnx.arange(10);     #rank-1 Tensor from [0,10) with step 1
    B = cytnx.arange(0,10,2); #rank-1 Tensor from [0,10) with step 2
    C = cytnx.ones([3,4,5]);  #Tensor of shape (3,4,5) with all elements set to one.
    D = cytnx.eye(3);         #Tensor of shape (3,3) with diagonal elements set to one, all other entries are zero.

* In C++:

.. code-block:: c++
    :linenos:

    auto A = cytnx::arange(10);     //rank-1 Tensor from [0,10) with step 1
    auto B = cytnx::arange(0,10,2); //rank-1 Tensor from [0,10) with step 2
    auto C = cytnx::ones({3,4,5});  //Tensor of shape (3,4,5) with all elements set to one.
    auto D = cytnx.eye(3);          //Tensor of shape (3,3) with diagonal elements set to one, all other entries are zero.


:Tips: In C++, you can make use of *auto* to simplify your code! 

Random Tensor
************************
Often, Tensors shall be initialized with random values. This can be achieved with **random.normal** (normal or Gaussian distribution) and **random.uniform** (uniform distribution):

* In Python : 

.. code-block:: python 
    :linenos:

    A = cytnx.random.normal([3,4,5], mean=0., std=1.)   #Tensor of shape (3,4,5) with all elements distributed according
                                                        #to a normal distribution around 0 with standard deviation 1
    B = cytnx.random.uniform([3,4,5], low=-1., high=1.) #Tensor of shape (3,4,5) with all elements distributed uniformly
                                                        #between -1 and 1
    

* In C++:

.. code-block:: c++
    :linenos:

    auto A = cytnx::random::normal({3,4,5}, 0., 1.);    //Tensor of shape (3,4,5) with all elements distributed according
                                                        //to a normal distribution around 0 with standard deviation 1
    auto B = cytnx::random::uniform({3,4,5}, -1., 1.);  //Tensor of shape (3,4,5) with all elements distributed uniformly
                                                        //between -1 and 1


Tensor with different dtype and device 
*******************************************
By default, a Tensor will be created with elements of type *double* (or *float* in Python) on the CPU if there are no additional arguments provided upon creating the Tensor. 

You can create a Tensor with a different data type, and/or on different devices simply by specifying the **dtype** and the **device** arguments upon initialization. For example, the following code creates a Tensor with 64bit integer elements on a cuda-enabled GPU:

* In Python:

.. code-block:: python

    A = cytnx.zeros([3,4,5],dtype=cytnx.Type.Int64,device=cytnx.Device.cuda)

* In C++:

.. code-block:: c++

    auto A = cytnx.zeros({3,4,5},cytnx::Type.Int64,cytnx::Device.cuda);


.. Note:: 
    
    1. Remember to switch between '.' in Python and '::' in C++ when you use Type and Device classes. 
    
    2. If you have multiple GPUs, you can specify on which GPU you want to initialize the Tensor by adding the gpu-id to cytnx::Device::cuda. 
        
        For example: 
        
            device=cytnx.Device.cuda+2   #will create the Tensor on GPU id=2

            device=cytnx.Device.cuda+4   #will create the Tensor on GPU id=4

    3. In C++, there are no keyword arguments as Python, so make sure you put the arguments in the correct order. Check the `API documentation <https://kaihsin.github.io/Cytnx/docs/html/index.html>`_ for function signatures!  


Currently, there are several data types supported by Cytnx (see :ref:`Scalar` for more details):

.. tabularcolumns:: |l|l|l|

+------------------+----------------------+-------------------+
| Cytnx type       | C++ type             | Type object       |
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


Concerning devices, Cytnx currently supports

.. tabularcolumns:: |l|l|

+------------------+----------------------+
| device           | Device object        |
+==================+======================+
| CPU              | Device.cpu           | 
+------------------+----------------------+
| CUDA-enabled GPU | Device.cuda+x        |
+------------------+----------------------+

Type conversion 
**********************
It is possible to convert a Tensor to a different data type. To convert the data type, simply use **Tensor.astype()**.

For example, consider a Tensor *A* with **dtype=Type.Int64**, which shall be converted to **Type.Double**:

* In Python:

.. code-block:: python 
    :linenos:
    
    A = cytnx.ones([3,4],dtype=cytnx.Type.Int64)
    B = A.astype(cytnx.Type.Double)
    print(A.dtype_str())
    print(B.dtype_str())

* In C++:

.. code-block:: c++
    :linenos:

    auto A = cytnx::ones({3,4},cytnx::Type.Int64);
    auto B = A.astype(cytnx::Type.Double);
    cout << A.dtype_str() << endl;
    cout << B.dtype_str() << endl;

>> Output:

.. code-block:: text
    
    Int64
    Double (Float64)



.. Note::
    
    1. Tensor.dtype() returns a type-id, while Tensor.dtype_str() returns the type name. 
    2. A complex data type cannot directly be converted to a real data type. Use Tensor.real() or Tensor.imag() if you want to get the real or imaginary part.


Transfer between devices
***************************
Moving a Tensor between different devices is very easy. We can use **Tensor.to()** to move the Tensor to a different device.

For example, let's create a Tensor in the memory accessible by the CPU and transfer it to the GPU with gpu-id=0. 

* In Python:

.. code-block:: python 
    :linenos:

    A = cytnx.ones([2,2]) #on CPU
    B = A.to(cytnx.Device.cuda+0)
    print(A) # on CPU
    print(B) # on GPU

    A.to_(cytnx.Device.cuda) 
    print(A) # on GPU


* In C++:

.. code-block:: c++
    :linenos:

    auto A = cytnx::ones([2,2]); //on CPU
    auto B = A.to(cytnx::Device.cuda+0);
    cout << A << endl; // on CPU
    cout << B << endl; // on GPU
    
    A.to_(cytnx::Device.cuda);
    cout << A << endl; // on GPU


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

    Total elem: 4
    type  : Double (Float64)
    cytnx device: CUDA/GPU-id:0
    Shape : (2,2)
    [[1.00000e+00 1.00000e+00 ]
     [1.00000e+00 1.00000e+00 ]]


.. Note::
    
    1. You can use **Tensor.device()** to get the current device-id (cpu = -1), whereas **Tensor.device_str()** returns the device name. 

    2. **Tensor.to()** will return a copy on the target device. If you want to move the current Tensor to another device, use **Tensor.to_()** (with underscore). 


Tensor from Storage [v0.6.6+]
*******************************
    The Storage of a tensor contains the actual tensor elements. They are stored in the form of a vector. Further details about Storage objects are explained in :ref:`Storage`.  

    We can create a Tensor directly from a Storage object by using **Tensor.from_storage()**: 

* In Python:

.. code-block:: python
    :linenos:

    ## A & B share same memory
    A = cytnx.Storage(10);
    B = cytnx.Tensor.from_storage(A);

    ## A & C have different memory
    C = cytnx.Tensor.from_storage(A.clone());

    
* In C++:

.. code-block:: c++
    :linenos:

    // A & B share same memory
    auto A = cytnx::Storage(10);
    auto B = cytnx::Tensor::from_storage(A);

    // A & C have different memory
    C = cytnx::Tensor::from_storage(A.clone());


.. Note::

    Note that this will create a wrapping of the Storage in a Tensor. The created Tensor and the input storage share the same memory. To create independent memory for the Tensor data, use **storage.clone()**


.. toctree::
