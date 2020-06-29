Append elements
-----------------
The size of Tensor can be expanded using **Tensor.append**. 

One can append a scalar in to a rank-1 Tensor.
For example:

* In python:

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
    
   It is not possible to append a scalar into a Tensor with rank>1, as this operation is by itself ambiguous.

For Tensor with rank>1, you can append a Tensor into it, provided the shape is matching. This operation is equivalent as `numpy.vstack <https://numpy.org/doc/stable/reference/generated/numpy.vstack.html>`_.

For example, consider a Tensor with shape (3,4,5), you can append a Tensor with shape (4,5) into it, and the resulting output will be in shape (4,4,5).

* In python:

.. code-block:: python
    :linenos:

    A = cytnx.ones([3,4,5])
    B = cytnx.ones([4,5])*2
    print(A)
    print(B)

    A.append(B)
    print(A)


.. code-block:: c++
    :linenos:

    auto A = cytnx::ones({3,4,5});
    auto B = cytnx.ones({4,5})*2;
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
    
    You cannot append a complex type scalar/Tensor into a real type Tensor. 




.. toctree::
