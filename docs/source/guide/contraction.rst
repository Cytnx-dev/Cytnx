Contraction
=============

Tensor contractions are the basis of tensor network algorithms: tensors are multiplied and indices shared by two tensors are summed over. Cytnx provides three different ways to do these contractions.

The most advanced method for tensor contractions is a **Network** object. It allows to define a tensor network and its connectivity. This provides an abstract description of the tensor content and the open and contracted bonds. One can apply this mask to a concrete set of tensors and let Cytnx perform the contractions. It is possible to define the index order of the contractions, or to let Cytnx automatically find an optimal sequence. The network can be reused for a different set of tensors, and also the optimized contraction order can be reused once calculated. The bonds of the tensors to be loaded can either be called by their labels or index order. A network is most conveniently defined in a file which can be read by Cytnx.

A simple way to contract indices is provided by **Contract()**. This functions contract all indices with the same labels on two or more tensors.

Finally, the function **ncon()** allows to contract tensors by defining the connectivity and contraction order of the bonds. The user needs to specify the bonds by their indices instead of their labels, so the index order matters.


.. Tip::

    We encourage users to use **Network** contractions or **Contract** together with meaningful label names, which are easier to understand than integer values as required by **ncon**. The latter routine is provided for advanced users who carefully track the index order themselves.



.. toctree::
    :maxdepth: 3

    contraction/network.rst
    contraction/contract.rst
    contraction/ncon.rst
