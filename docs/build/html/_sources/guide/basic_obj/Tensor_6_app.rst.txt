Appending elements
-------------------
The size of a Tensor can be expanded using **Tensor.append**. 

One can append a scalar to a rank-1 Tensor.
For example:

* In Python: 

.. code-block:: python
    :linenos:

    A = cytnx.ones(4)
    print(A)
    A.append(4)
    print(A)

* In C++:

.. code-block:: c++
    :linenos:

    auto A = cytnx::ones(4);
    cout << A << endl;
    A.append(4);
    cout << A << endl;

Output>> 

.. code-block:: text

    Total elem: 4
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (4)
    [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]


    Total elem: 5
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (5)
    [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 4.00000e+00 ]


.. Note::
    
   It is not possible to append a scalar to a Tensor with rank > 1, as this operation is by itself ambiguous.

For Tensors with rank > 1, you can append a Tensor to it, provided the shape is matching. This operation is equivalent to :numpy-vstack:`numpy.vstack <>`.

For example, consider a Tensor with shape (3,4,5). You can append a Tensor with shape (4,5) to it, and the resulting output will have shape (4,4,5).

* In Python:

.. code-block:: python
    :linenos:

    A = cytnx.ones([3,4,5])
    B = cytnx.ones([4,5])*2
    print(A)
    print(B)

    A.append(B)
    print(A)

* In C++:
.. code-block:: c++
    :linenos:

    auto A = cytnx::ones({3,4,5});
    auto B = cytnx::ones({4,5})*2;
    cout << A << endl;
    cout << B << endl;

    A.append(B);
    cout << A << endl;

Output>>

.. code-block:: text

    Total elem: 60
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (3,4,5)
    [[[1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]
      [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]
      [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]
      [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]]
     [[1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]
      [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]
      [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]
      [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]]
     [[1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]
      [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]
      [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]
      [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]]]


    Total elem: 20
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (4,5)
    [[2.00000e+00 2.00000e+00 2.00000e+00 2.00000e+00 2.00000e+00 ]
     [2.00000e+00 2.00000e+00 2.00000e+00 2.00000e+00 2.00000e+00 ]
     [2.00000e+00 2.00000e+00 2.00000e+00 2.00000e+00 2.00000e+00 ]
     [2.00000e+00 2.00000e+00 2.00000e+00 2.00000e+00 2.00000e+00 ]]


    Total elem: 80
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (4,4,5)
    [[[1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]
      [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]
      [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]
      [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]]
     [[1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]
      [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]
      [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]
      [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]]
     [[1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]
      [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]
      [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]
      [1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 1.00000e+00 ]]
     [[2.00000e+00 2.00000e+00 2.00000e+00 2.00000e+00 2.00000e+00 ]
      [2.00000e+00 2.00000e+00 2.00000e+00 2.00000e+00 2.00000e+00 ]
      [2.00000e+00 2.00000e+00 2.00000e+00 2.00000e+00 2.00000e+00 ]
      [2.00000e+00 2.00000e+00 2.00000e+00 2.00000e+00 2.00000e+00 ]]]

.. Note::
    
    1. The Tensor to be appended must have the same shape as the Tensor to append to, but with one index (the first one) less.  
    2. You cannot append a complex type scalar/Tensor to a real type Tensor. 


.. toctree::
