linalg extension
==================

.. .. toctree::
..     :maxdepth: 3

Tensor decomposition
**************************



As mention in the **Manipulate UniTensor**, the specification of **rowrank** makes it convinient to apply linear algebra operations on UniTensors.


Singular value decomposition
-------------------------------

Here is an example where a **singular value decomposition (SVD)** is performed on a UniTensor:

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


Here we demonstrate the usage of a more important function **Svd_truncate()** which appears frequently in the tensor network algorithm for truncatiing the bond dimension. In this example we print the singular values from doing **Svd()** and compare it to the result of **Svd_truncate()**:

* In Python:

.. literalinclude:: ../../code/python/doc_codes/guide_xlinalg_Svd_truncate.py
    :language: python
    :linenos:


Output >>

.. literalinclude:: ../../code/python/outputs/guide_xlinalg_Svd_truncate.out
    :language: text

We note that the singular values obtained by doing **Svd_truncate()** is truncated according to our **err** requirment, a **keepdim** argument is also passed to specify a maximum desired dimension. Finally we note that by setting **return_err = 1** we can get the largest truncated singular values, it is also possible to obtain all truncated value by passing any int \>1.


.. Note::
    The **Svd()** and **Svd_truncate()** calls xgesdd routines from LAPACKE internally, for other algorithms like xgesvd routines one may consider  **Gesvd()** and **Gesvd_truncate()** functions.
    


Eigenvalue decomposition
-------------------------------

* In Python:

.. literalinclude:: ../../code/python/doc_codes/guide_xlinalg_Eig.py
    :language: python
    :linenos:

QR decomposition
-------------------------------

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