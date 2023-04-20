Save/Load
-----------------
We can also save/read Tensors to/from a file.

Save a Tensor
*****************
To save a Tensor to a file, simply call **Tensor.Save(filepath)**. 

* In python:

.. code-block:: python 
    :linenos:
    
    A = cytnx.arange(12).reshape(3,4)
    A.Save("T1")


* In c++:

.. code-block:: c++
    :linenos:

    auto A = cytnx::arange(12).reshape(3,4);
    A.Save("T1")

This will save Tensor *A* to the current directory as **T1.cytn**, with extension *.cytn*


Load a Tensor
******************
Now, let's load the Tensor from the file. 

* In python:

.. code-block:: python
    :linenos:
    
    A = cytnx.Tensor.Load("T1.cytn")
    print(A)

* In c++:

.. code-block:: c++
    :linenos:
    
    auto A = cytnx::Tensor::Load("T1.cytn");
    cout << A << endl;

Output>>

.. code-block:: text

    Total elem: 12
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (3,4)
    [[0.00000e+00 1.00000e+00 2.00000e+00 3.00000e+00 ]
     [4.00000e+00 5.00000e+00 6.00000e+00 7.00000e+00 ]
     [8.00000e+00 9.00000e+00 1.00000e+01 1.10000e+01 ]]


.. toctree::
