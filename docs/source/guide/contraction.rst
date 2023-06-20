Contraction
=============

All Tensor Network algorithms include Tensor contractions, where we multiply tensors and sum over shared indices. Cytnx provides three different ways to do these contractions.

The most advanced method for tensor contractions is a **Network** object. It allows to define a Tensor Network and its connectivity. This provides an abstract description of the tensor content and the open and contracted bonds. One can apply this mask to a concrete set of tensors and let Cytnx calculate the contractions. It is possible to define the index order of the contractions, or to let Cytnx find the optimal sequence. The network can be reused for a different set of tensors, and also the optimized contraction order can be reused once calculated. The bonds of the tensors to be loaded can either be called by their labels or indices. A network is most conveniently defined in a file which can be read by Cytnx.

A simple way to contract indices is provided by **Contract()** and **Contracts()**. These functions contract all indices with the same labels on two or more tensors.

Finally, the function **ncon()** allows to contract tensors by defining the connectivity and contraction order of the bonds. The user needs to specify the bonds by their indices instead of their labels, so the index order matters.


.. toctree::
    :maxdepth: 3

    contraction/network.rst
    contraction/contract.rst
    contraction/ncon.rst