UniTensor
=================
Besides the generic tensor structure, Cytnx provides further objects that are designed specifically for tensor network simulations, based on the **cytnx.Tensor** class.

**UniTensor** is an enhanced version of **cytnx.Tensor** which provides features such as labels for each index (which we call "bond") and Bra/Ket (In/Out) convention of bonds. These features are handy for contractions of multiple tensors, as well as for the implementation of Tensor network diagrams. 

In the following, let's look into these objects: 

.. toctree::
    :maxdepth: 3

    cyx/TNotation.rst
    cyx/Bond.rst
    cyx/UniTensor.rst
    cyx/Contraction.rst

