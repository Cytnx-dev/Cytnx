From c++.vector
--------------------
Cytnx provide a way you can convert c++ *vector* directly to Storage. Simply use **Storage.from_vector**.

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

    cytnx overload the **operator<<** for c++ vector, you can directly print any vector when **using namespace cytnx;**.  


.. toctree::
