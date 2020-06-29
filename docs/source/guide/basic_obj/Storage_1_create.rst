Create a Storage
-------------------
The storage can be created in a similar way as in Tensor. Note that Storage does not have the concept of *shape*, and basically just like **vector** in C++.

To create a Storage, with dtype=Type.Double on cpu, 

* In python:

.. code-block:: python
    :linenos:

    A = cytnx.Storage(10,dtype=cytnx.Type.Double,device=cytnx.Device.cpu)
    A.set_zeros();

    print(A);

* In c++:

.. code-block:: c++
    :linenos:

    auto A = cytnx::Storage(10,cytnx::Type.Double,cytnx::Device.cpu);
    A.set_zeros();
    
    cout << A << endl;
    
Output>>

.. code-block:: text

    dtype : Double (Float64)
    device: cytnx device: CPU
    size  : 10
    [ 0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 ]


.. Note::
    
    Storage by itself only allocate memory (using malloc) without initialize it's elements. 


.. Tip::

    1. Use **Storage.set_zeros()** or **Storage.fill()** if you want to set all the elements to zero or some arbitrary numbers. 
    2. For complex type Storage, you can use **.real()** and **.imag()** to get the real part/imaginary part of the data. 



Type conversion
****************
Conversion between different data type is possible for Storage. Just like Tensor, call **Storage.astype()** to convert in between different data types. 

The available data types are the same as Tensor. 

* In python:

.. code-block:: python 
    :linenos:

    A = cytnx.Storage(10)
    A.set_zeros()

    B = A.astype(cytnx.Type.ComplexDouble)

    print(A)
    print(B)

* In c++:
 
.. code-block:: c++
    :linenos:
    
    auto A = cytnx::Storage(10);
    A.set_zeros();

    auto B = A.astype(cytnx::Type.ComplexDouble);

    cout << A << endl;
    cout << B << endl;

Output >>

.. code-block:: text

    dtype : Double (Float64)
    device: cytnx device: CPU
    size  : 10
    [ 0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 0.00000e+00 ]


    dtype : Complex Double (Complex Float64)
    device: cytnx device: CPU
    size  : 10
    [ 0.00000e+00+0.00000e+00j 0.00000e+00+0.00000e+00j 0.00000e+00+0.00000e+00j 0.00000e+00+0.00000e+00j 0.00000e+00+0.00000e+00j 0.00000e+00+0.00000e+00j 0.00000e+00+0.00000e+00j 0.00000e+00+0.00000e+00j 0.00000e+00+0.00000e+00j 0.00000e+00+0.00000e+00j  ]


Transfer btwn devices
***********************
We can also moving the storage between different devices. Just like Tensor, we can use **Storage.to()**. 

* In python:

.. code-block:: python
    :linenos:
    
    A = cytnx.Storage(4)
    B = A.to(cytnx.Device.cuda)

    print(A.device_str())
    print(B.device_str())

    A.to_(cytnx.Device.cuda)
    print(A.device_str())


* In c++:

.. code-block:: c++
    :linenos:

    auto A = cytnx::Storage(4);

    auto B = A.to(cytnx::Device.cuda);
    cout << A.device_str() << endl;
    cout << B.device_str() << endl;

    A.to_(cytnx::Device.cuda);
    cout << A.device_str() << endl;

Output>>

.. code-block:: text
    
    cytnx device: CPU
    cytnx device: CUDA/GPU-id:0
    cytnx device: CUDA/GPU-id:0


.. Hint::

    1. Like Tensor, **.device_str()** return the device string while **.device()** return device ID (cpu=-1).

    2. **.to()** return a copy on target device. Use **.to_()** instead to move the current instance to a target device. 


Get Storage of Tensor
**************************
Internally, data of Tensor is stored in Storage. We can get the storage of Tensor using **Tensor.storage()**. 

* In python:

.. code-block:: python 
    :linenos:

    A = cytnx.arange(10).reshape(2,5);
    B = A.storage();

    print(A)
    print(B)

* In c++:

.. code-block:: c++
    :linenos:

    auto A = cytnx::arange(10).reshape(2,5);
    auto B = A.storage();

    cout << A << endl;
    cout << B << endl;

Output >>

.. code-block:: text

    Total elem: 10
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (2,5)
    [[0.00000e+00 1.00000e+00 2.00000e+00 3.00000e+00 4.00000e+00 ]
     [5.00000e+00 6.00000e+00 7.00000e+00 8.00000e+00 9.00000e+00 ]]


    dtype : Double (Float64)
    device: cytnx device: CPU
    size  : 10
    [ 0.00000e+00 1.00000e+00 2.00000e+00 3.00000e+00 4.00000e+00 5.00000e+00 6.00000e+00 7.00000e+00 8.00000e+00 9.00000e+00 ]



.. Note::

    1. The return is the *referece* of Tensor's internal storage. That is, any modification to this storage will modify the Tensor accrodingly. 


**[Important]** For Tensor in non-contiguous status, since the meta-data is detached from it's memory that handled by storage, calling **Tensor.storage()** will return it's current memory layout, not the real view of Tensor. 

Let's use python API to demostrate this. The thing goes the with c++ API. 

* In python:

.. code-block:: python 
    :linenos:

    A = cytnx.arange(8).reshape(2,2,2)
    print(A.storage()) 

    # Let's make it non-contiguous 
    A.permute_(0,2,1)
    print(A.is_contiguous()) 

    # Note that the storage is not changed
    print(A.storage())

    # Now let's make it contiguous
    # thus the elements is moved
    A.contiguous_();
    print(A.is_contiguous())

    # Note that the storage now is changed 
    print(A.storage())
    

Output>>

.. code-block:: text

    dtype : Double (Float64)
    device: cytnx device: CPU
    size  : 8
    [ 0.00000e+00 1.00000e+00 2.00000e+00 3.00000e+00 4.00000e+00 5.00000e+00 6.00000e+00 7.00000e+00 ]

    False

    dtype : Double (Float64)
    device: cytnx device: CPU
    size  : 8
    [ 0.00000e+00 1.00000e+00 2.00000e+00 3.00000e+00 4.00000e+00 5.00000e+00 6.00000e+00 7.00000e+00 ]

    True

    dtype : Double (Float64)
    device: cytnx device: CPU
    size  : 8
    [ 0.00000e+00 2.00000e+00 1.00000e+00 3.00000e+00 4.00000e+00 6.00000e+00 5.00000e+00 7.00000e+00 ]



.. toctree::
