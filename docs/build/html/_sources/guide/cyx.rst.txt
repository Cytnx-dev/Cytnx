Cytnx extensions
=================
cytnx_extension submodule/namespace provides objects that are designed specifically for Tensor network simulations, base on the **cytnx.Tensor**.

The **cytnx_extension.CyTensor** which is a enhanced version of **cytnx.Tensor**, provides features such as labels for each rank (which is so called "bond"), Bra/Ket (In/Out) convention of bonds. These features are handy for multiple Tensor contraction, as well as implementation of Tensor network diagram. 

The **cytnx_extensioin.Network** object provides a skeleton of a Tensor network diagram. Users can write a *Network file*, which will serve as the blue print of a TN structure, and contract multiple Tensors at once with optimal contraction order optimized automatically within the **Network** object. Further more, user can plot the Tensor network diagram from *Network file* directly in python API to check the implementation. 

In the following, let's look into these objects: 


.. toctree::
    :maxdepth: 3

    cyx/TNotation.rst
    cyx/CyTensor.rst
    cyx/Network.rst
    cyx/Bond.rst


