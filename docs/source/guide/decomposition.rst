Tensor decomposition
=====================

.. .. toctree::
..     :maxdepth: 3

As mention in the :ref:`Manipulating a UniTensor`, the specification of **rowrank** makes it convenient to apply linear algebra operations on UniTensors.


Singular value decomposition
*****************************

Here is an example where a **singular value decomposition (SVD)** is performed on a UniTensor:

* In Python:

.. literalinclude:: ../../code/python/doc_codes/guide_xlinalg_Svd.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../code/python/outputs/guide_xlinalg_Svd.out
    :language: text


.. toctree::

When calling *Svd*, the first three legs of **T** are automatically reshaped into one leg according to **rowrank=3**. After the SVD, the matrices **U** and **Vt** are automatically reshaped back into the corresponding index form of the original tensor. This way, we can reconstruct the original UniTensor **T** by contracting :math:`U \cdot S \cdot Vt`:

* In Python:

.. literalinclude:: ../../code/python/doc_codes/guide_xlinalg_Svd_verify.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../code/python/outputs/guide_xlinalg_Svd_verify.out
    :language: text

If we contract :math:`U \cdot S \cdot Vt`, we get a tensor of the same shape as **T** and we can subtract the two tensors. The error :math:`\frac{|T-U \cdot S \cdot Vt|}{|T|}` is of the order of machine precision, as expected.

If the singular values have a decaying hierarchy, then the smallest values can be omitted without introducing a large error to the reconstructed tensor. More precisely, the 2-norm of the error between the original tensor and the reconstructed tensor is minimized. This truncated SVD is the basis of many tensor network algorithms.

Here we demonstrate the usage of **Svd_truncate()**, which implements this truncation. In this example we print the singular values obtained by **Svd()**, and compare them to the result of **Svd_truncate()**:

* In Python:

.. literalinclude:: ../../code/python/doc_codes/guide_xlinalg_Svd_truncate.py
    :language: python
    :linenos:


Output >>

.. literalinclude:: ../../code/python/outputs/guide_xlinalg_Svd_truncate.out
    :language: text

We note that the singular values obtained by **Svd_truncate()** are truncated according to our **err** requirement, a **keepdim** argument is also passed to specify a maximum desired dimension. Finally we note that by setting **return_err = 1** we can get the largest truncated singular values, it is also possible to obtain all truncated value by passing any int \>1.


.. Note::
    The **Svd()** and **Svd_truncate()** calls xgesdd routines from LAPACKE internally, for other algorithms like xgesvd routines one may consider  **Gesvd()** and **Gesvd_truncate()** functions.



Eigenvalue decomposition
*****************************

* In Python:

.. literalinclude:: ../../code/python/doc_codes/guide_xlinalg_Eig.py
    :language: python
    :linenos:

QR decomposition
*****************************

The **QR decomposition** decomposes a matrix *M* to the form *M = QR*, where *Q* is an orthogonal matrix (*Q Q^T = I*), and *R* is a upper-right triangular matrix. One can perform a QR decomposition by using **Qr()**.

.. py:function:: Qr(Tin, is_tau)

    :param cytnx.UniTensor Tin: input tensor
    :param bool is_tau: If *is_tau=True*, the function returns an additional one-dimensional tensor *tau* that contains the scaling factors of the Householder reflectors that generate *Q* along with *R*. See :cite:`LAPACK` for details. Default: *is_tau=False*

Here is an example of a QR decomposition:

* In Python:

.. literalinclude:: ../../code/python/doc_codes/guide_xlinalg_Qr.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../code/python/outputs/guide_xlinalg_Qr.out
    :language: text

.. bibliography:: ref.xlinalg.bib
    :cited:
