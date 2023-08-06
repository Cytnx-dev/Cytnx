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

.. literalinclude:: ../../../code/cplusplus/guide_codes/3_6_ex1.cpp
    :language: c++
    :linenos:

Output>> 

.. literalinclude:: ../../../code/cplusplus/outputs/3_6_ex1.out
    :language: text

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
.. literalinclude:: ../../../code/cplusplus/guide_codes/3_6_ex2.cpp
    :language: c++
    :linenos:

Output>>

.. literalinclude:: ../../../code/cplusplus/outputs/3_6_ex2.out
    :language: text

.. Note::
    
    1. The Tensor to be appended must have the same shape as the Tensor to append to, but with one index (the first one) less.  
    2. You cannot append a complex type scalar/Tensor to a real type Tensor. 


.. toctree::
