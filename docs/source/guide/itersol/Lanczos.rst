Lanczos solver
----------------
In Cytnx, an eigenvalue problem can be solved for a custom linear operator defined using the **LinOp** class by using the Lanczos iterative solver.

For this, you can pass a **LinOp** or any of its child classes to **linalg.Lanczos_ER**:

.. py:function:: Lanczos_ER(Hop, k=1, is_V=true, maxiter=10000,\
                CvgCrit=1.0e-14, is_row=false, Tin=Tensor(), max_krydim=4)

    Perform Lanczos for hermitian/symmetric matrices or linear function.

    :param LinOp Hop: the Linear Operator defined by LinOp class or its inheritance.
    :param uint64 k: the number of lowest k eigenvalues
    :param bool is_V: if set to true, the eigenvectors will be returned
    :param uint64 maxiter: the maximum number of iteration steps for each k
    :param double CvgCrit: the convergence criterion of the energy
    :param bool is_row: whether the returned eigenvectors should be in row-major form
    :param Tensor Tin: the initial vector, should be a Tensor with rank-1
    :param uint32 max_krydim: the maximum Krylov subspace dimension for each iteration
    :return: [eigvals (Tensor), eigvecs (Tensor)(option)]
    :rtype: vector<Tensor> (C++ API)/list of Tensor(python API)


For example, we consider a simple example where we wrap a (4x4) matrix inside a custom operator. We can easily generalize the **matvec** to be any custom sparse structure.

* In Python:

.. literalinclude:: ../../../code/python/doc_codes/guide_itersol_Lanczos_Lanczos.py
    :language: python
    :linenos:

* In C++:

.. literalinclude:: ../../../code/cplusplus/doc_codes/guide_itersol_Lanczos_Lanczos.cpp
    :language: c++
    :linenos:

Output >>

.. literalinclude:: ../../../code/python/outputs/guide_itersol_Lanczos_Lanczos.out
    :language: text


.. Note::

    1. The ER stand for explicitly restarted. The Lanczos method used is based on :lanczos-er:`this reference <>` which can reproduce the degenerate correctly.

    2. The Lanczos solver only works for symmetric/Hermitian operators.

    3. In cases where the operator is small, try to reduce the max_krydim to get a correct convergence.

.. seealso::

    The solver is used in the example :ref:`SectionED` for the one dimensional transverse field Ising model.
