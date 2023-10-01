linalg extension
==================

.. .. toctree::
..     :maxdepth: 3

Tensor decomposition
**************************

As mention in the **Manipulate UniTensor**, the specification of **rowrank** makes it convinient to apply linear algebra operations on UniTensors. Here is an example where a **singular value decomposition (SVD)** is performed on a UniTensor:

* In Python:

.. literalinclude:: ../../code/python/doc_codes/guide_xlinalg_Svd.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../code/python/outputs/guide_xlinalg_Svd.out
    :language: text


.. toctree::

When calling *Svd*, the first three legs of **T** are automatically reshaped into one leg according to **rowrank=3**. After the SVD, the matrices **U** and **Vt** are automatically reshaped back into the corresponding index form of the original tensor. This way, we get the original UniTensor **T** if we contract :math:`U \cdot S \cdot Vt`:

* In Python:

.. literalinclude:: ../../code/python/doc_codes/guide_xlinalg_Svd_verify.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../code/python/outputs/guide_xlinalg_Svd_verify.out
    :language: text

If we contract :math:`U \cdot S \cdot Vt`, we get a tensor of the same shape as **T** and we can subtract the two tensors. The error :math:`\frac{|T-U \cdot S \cdot Vt|}{|T|}` is of the order of machine precision, as expected.
