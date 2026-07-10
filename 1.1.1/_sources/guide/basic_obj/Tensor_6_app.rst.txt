Appending elements
-------------------
The size of a Tensor can be expanded using **Tensor.append**.

One can append a scalar to a rank-1 Tensor.
For example:

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_basic_obj_Tensor_6_app_scalar.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_basic_obj_Tensor_6_app_scalar.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_basic_obj_Tensor_6_app_scalar.out
    :language: text

.. Note::

   It is not possible to append a scalar to a Tensor with rank > 1, as this operation is by itself ambiguous.

For Tensors with rank > 1, you can append a Tensor to it, provided the shape is matching. This operation is equivalent to :numpy-vstack:`numpy.vstack <>`.

For example, consider a Tensor with shape (3,4,5). You can append a Tensor with shape (4,5) to it, and the resulting output will have shape (4,4,5).

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_basic_obj_Tensor_6_app_tensor.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_basic_obj_Tensor_6_app_tensor.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_basic_obj_Tensor_6_app_tensor.out
    :language: text

.. Note::

    1. The Tensor to be appended must have the same shape as the Tensor to append to, but with one index (the first one) less.
    2. You cannot append a complex type scalar/Tensor to a real type Tensor.


.. toctree::
