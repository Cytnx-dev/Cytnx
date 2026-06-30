Iterative solver
==================
In addition to offering basic linear algebra functions for dense tensors, Cytnx also includes an iterative solver for solving eigenvalue problems. By using the LinOp class, users can define a custom linear operator as a function or class, similar to
:spLO:`scipy.LinearOperator <>`.

This approach is more versatile than using the deprecated sparse storage format. Linear algebra algorithms can be implemented matrix-free with a LinOp object that defines how an input tensor is transformed, for example by contracting it in a more complicated network.

In the following section, we demonstrate how to define a custom linear operator using the **LinOp** class and how to solve the eigenvalue problem using the iterative solver.


.. toctree::
    :maxdepth: 3

    itersol/LinOp.rst
    itersol/Lanczos.rst
