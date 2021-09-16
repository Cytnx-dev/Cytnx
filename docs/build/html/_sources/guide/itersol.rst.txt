Iterative solver
==================
Beside the basic linear algebra functions for direct solving of dense Tensor, Cytnx also provides the iterative solver that solve the eigen value problem. Together with LinOp class, user can define a custom linear operator as a function/class, similar as 
:spLO:`scipy.LinearOperator <>`.

This is more general than the conventional case of using the standard sparse storage format.

 
In the following, we show how one can define a customize linear operator using **LinOp** class, and solving the eigen value problem using iterative solver.  


.. toctree::
    :maxdepth: 3

    itersol/LinOp.rst
    itersol/Lanczos.rst
