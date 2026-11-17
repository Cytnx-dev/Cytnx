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

Choosing the driver of Eigh
---------------------------

**Eigh()** and **Eigh_truncate()** take an **algorithm** argument that selects the routine computing the decomposition:

* ``"qr"``: QR iteration (LAPACK ``?syev`` / ``?heev``). It is available on the CPU only.
* ``"dc"``: divide and conquer (LAPACK ``?syevd`` / ``?heevd``; cuSOLVER ``?syevd`` / ``?heevd`` on a GPU).
* ``"default"``: ``"qr"`` on the CPU and ``"dc"`` on a GPU. This is what **Eigh()** did before it had this argument, so a call that does not name an algorithm is unchanged.

An unknown name, or a driver that the device of the input does not have, raises an error. For a UniTensor with symmetries the same driver is used in every charge sector.

**A faster Eigh on the CPU.** On the CPU the default is QR iteration, which is the slower driver when eigenvectors are requested. Divide and conquer, the driver that **Eigh()** on a GPU and ``numpy.linalg.eigh`` use, has to be asked for:

* In Python:

.. code-block:: python

    e, V = cytnx.linalg.Eigh(H, algorithm="dc")
    out = cytnx.linalg.Eigh_truncate(rho, keepdim=20, algorithm="dc")

* In C++:

.. code-block:: c++

    auto out = cytnx::linalg::Eigh(H, true, false, "dc");

What ``"dc"`` changes, compared with the CPU default:

* **Time.** With eigenvectors, ``"dc"`` is usually faster once the matrix has a few hundred rows, and the gain grows with the size. How large it is depends on the LAPACK library. Measured on one Linux machine with 8 threads, for a 1250 x 1250 matrix: 6.6 times faster (Double) and 9.7 times faster (ComplexDouble) with OpenBLAS, and 2.3 times and 1.7 times faster with MKL. It is not faster in every case: a 450 x 450 ComplexDouble matrix took 17% longer with ``"dc"`` on OpenBLAS. Without eigenvectors (``is_V=False``) the two drivers take the same time. Measure on your own matrices if the time matters.
* **Memory.** With eigenvectors, divide and conquer needs a workspace of about :math:`2 n^2` numbers of the dtype of the matrix for an :math:`n \times n` matrix, on top of the matrix itself; QR iteration needs about :math:`3 n`. For a Double matrix with :math:`n = 8000` that is about 1 GB more.
* **Eigenvalues.** The same up to rounding, that is, up to a few units in the last place of the largest eigenvalue magnitude.
* **Eigenvectors.** Equally valid, but not the same numbers: each vector can come back with a different sign (phase), and inside a degenerate eigenspace the vectors form a different orthonormal basis. See the note below. :math:`V D V^\dagger` and the projector onto each eigenspace are the same up to rounding.
* **Eigh_truncate().** The kept eigenvalues are the same up to rounding. An exact degeneracy at the cut is kept entirely with either driver; where rounding splits a multiplet that the cut falls into, the vectors that are kept depend on the driver.

The argument applies to the call it is passed to. Functions that call **Eigh()** internally, such as **ExpH()** and **Lanczos()** with ``method="ER"`` (for its small Krylov matrix), use the default.

If LAPACK reports that divide and conquer did not converge, **Eigh()** prints a warning and redoes the decomposition with QR iteration.

.. note::

    The eigenvalues, and the projector onto each eigenspace, are the same for every **algorithm** up to rounding. The eigenvectors are not: their signs (phases) and the basis they form inside a degenerate eigenspace are not fixed by the decomposition, and they differ between drivers and between the CPU and a GPU. Code that compares :math:`V` entry by entry with stored eigenvectors, or with the eigenvectors from another driver or device, will see differences; compare :math:`V D V^\dagger`, not :math:`V`.

Truncated Hermitian eigenvalue decomposition
--------------------------------------------

For a Hermitian matrix, **Eigh()** returns the eigenvalues in ascending order; on a UniTensor with symmetries it does so block by block, one block per charge sector. **Eigh_truncate()** is the counterpart of **Svd_truncate()** for this case: it keeps the **keepdim** eigenpairs of largest magnitude :math:`|\lambda|`, ranked across all blocks, and returns the truncated eigenvalue UniTensor, the eigenvectors with a truncated eigen bond, and, with **return_err**, the discarded magnitudes. The arguments **keepdim**, **err**, **mindim**, **min_blockdim** and **return_err** have the same meaning as in **Svd_truncate()**. A typical use is the truncation of a reduced density matrix to its dominant eigenbasis, sector by sector, in one call.

* In Python:

.. literalinclude:: ../../code/python/doc_codes/guide_xlinalg_Eigh_truncate.py
    :language: python
    :linenos:

Output >>

.. literalinclude:: ../../code/python/outputs/guide_xlinalg_Eigh_truncate.out
    :language: text

The ranking is global: eigenvalue magnitudes are compared across all sectors. Without per-block floors that change this selection, the block-sparse and dense paths keep the same eigenvalue magnitudes. When the retained count is exactly **keepdim**, the reconstruction :math:`V D V^\dagger` is an optimal rank-**keepdim** approximation in spectral norm (the Hermitian case of the Eckart-Young theorem). The **err** cutoff can reduce the retained count; per-block floors or an exact magnitude degeneracy at the cut can increase it. Floors constrain the selection and can exclude the unconstrained dense optimum: sectors with spectra ``[-3, 0.5, 5]`` and ``[1, 4]``, **keepdim = 2** and floors ``[0, 2]`` retain ``[1, 4]`` (spectral error 5), whereas the unconstrained optimum retains ``[5, 4]`` (error 3). **return_err** reports the magnitudes actually discarded, with mode 1 reporting their maximum (the spectral-norm reconstruction error).

For a dense UniTensor, the one-element **min_blockdim** acts as **mindim** and prevents cutoff-based removal; it does not override **keepdim**. For block inputs, floors are reserved before applying the global budget. Consequently **keepdim = 1**, **min_blockdim = [3]** retains one eigenpair for a dense input but three for a one-sector block input with at least three eigenpairs. Dense eigenvectors also inherit **Eigh()**'s flattened row space and rowrank 1; block eigenvectors preserve the row-space bonds and rowrank. For example a dense shape ``[1, 3, 1, 3]`` with rowrank 2 returns eigenvectors of shape ``[3, 2]`` with rowrank 1 when two eigenpairs are kept.

Dtypes follow **Eigh()**. For fermionic inputs, materializing pending sign flips can promote a real Float sector to Double; truncation inherits the dtype of the selected sectors.

.. Note::
    **Eigh_truncate()** selects by magnitude, not by algebraic value. To select eigenpairs by value, for instance the lowest or the highest eigenvalues of a Hamiltonian, call **Eigh()** and keep the wanted part of its ascending spectrum (for a symmetric UniTensor, block by block), or use **Lanczos()** with **which = "SA"** (smallest algebraic) or **"LA"** (largest algebraic), which computes only the wanted eigenpairs.

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
