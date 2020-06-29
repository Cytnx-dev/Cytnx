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

    auto A = cytnx::Storage(10,cytnx::Type::Double,cytnx::Device::cpu);
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

    auto B = A.astype(cytnx::Type::ComplexDouble);

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




.. toctree::
