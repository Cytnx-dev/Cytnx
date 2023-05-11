UniTensor
=================
Cytnx provides objects that are designed specifically for Tensor network simulations, base on the **cytnx.Tensor**.

The **UniTensor** which is a enhanced version of **cytnx.Tensor**, provides features such as labels for each rank (which is so called "bond"), Bra/Ket (In/Out) convention of bonds. These features are handy for multiple Tensor contraction, as well as implementation of Tensor network diagram. 

In the following, let's look into these objects: 

.. toctree::
    :maxdepth: 3

    cyx/TNotation.rst
    cyx/Bond.rst
    cyx/UniTensor.rst
    cyx/Contraction.rst

