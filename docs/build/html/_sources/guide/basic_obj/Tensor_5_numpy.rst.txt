To/From numpy.array
----------------------
Cytnx also provides conversion from and to numpy.array in the Python API. 

* To convert from Tensor to numpy array, use **Tensor.numpy()**

.. code-block:: python
    :linenos:
        
        A = cytnx.ones([3,4])
        B = A.numpy()
        print(A)
        print(type(B))
        print(B)

Output>>

.. code-block:: text

    Total elem: 12
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (3,4)
    [[1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]
     [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]
     [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]]


    <class 'numpy.ndarray'>
    [[1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]]



* To convert from numpy array to Tensor, use **cytnx.from_numpy()**

.. code-block:: python 
    :linenos:

        import numpy as np
        B = np.ones([3,4])
        A = cytnx.from_numpy(B)
        print(B)
        print(A)

Output>>

.. code-block:: text

    [[1. 1. 1. 1.]
     [1. 1. 1. 1.]
     [1. 1. 1. 1.]]

    Total elem: 12
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (3,4)
    [[1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]
     [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]
     [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]]


.. toctree::
