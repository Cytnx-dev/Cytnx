From/To c++.vector
--------------------
Cytnx provide a way you can convert c++ *vector* directly to and from Storage. 


To convert a c++ vector to Storage, using **Storage::from_vector**:

* In C++

.. code-block:: c++
    :linenos:

    vector<double> vA(4,6);
    
    auto A = cytnx::Storage::from_vector(vA);
    auto B = cytnx::Storage::from_vector(vA,cytnx::Device.cuda);

    cout << A << endl;
    cout << B << endl;

Output >>

.. code-block:: text

    dtype : Double (Float64)
    device: cytnx device: CPU
    size  : 4
    [ 6.00000e+00 6.00000e+00 6.00000e+00 6.00000e+00 ]

    dtype : Double (Float64)
    device: cytnx device: CUDA/GPU-id:0
    size  : 4
    [ 6.00000e+00 6.00000e+00 6.00000e+00 6.00000e+00 ]

.. Note::

    You can also specify the device upon calling *from_vector*. 

.. Tip::

    Cytnx overload the **operator<<** for c++ vector, you can directly print any vector when **using namespace cytnx;**.  
    Alternatively, you can also use **print()** function just like in python.



(new v0.7.5+)
To convert a Storage to std::vector with type *T*, using **Storage.vector<T>()**:


* In C++

.. code-block:: c++
    :linenos:

    Storage sA = {3.,4.,5.,6.};

    print(sA.dtype_str());
    
    auto vA = sA.vector<double>();

    print(vA);
    
Output >>

.. code-block:: text

    Double (Float64)

    Vector Print:
    Total Elements:4
    [3, 4, 5, 6]

.. Note::

    The type T has to match the dtype of Storage, otherwise an error will raise. 




.. toctree::
