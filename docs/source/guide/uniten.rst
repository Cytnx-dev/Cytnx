UniTensor
=================
Besides the generic tensor structure, Cytnx provides further objects that are designed specifically for tensor network simulations, based on the **cytnx.Tensor** class.

**UniTensor** is an enhanced version of **cytnx.Tensor** which provides features such as labels for each index (which we call "bond") and Bra/Ket (In/Out) convention of bonds. These features are handy for contractions of multiple tensors, as well as for the implementation of Tensor network diagrams. 

A **UniTensor** consist of three important parts: 

.. image:: uniten/image/UT.png
    :width: 600
    :align: center


1. Block(s), which is a **cytnx.Tensor** (or a list/vector thereof). Block(s) store all the elements of the tensor. 

2. Bonds, which contain the information associated to each index of the tensor. 

3. labels, which give each bond a name with which it can be accessed. 



In the simplest case where no symmetries of the system are considered, a **UniTensor** can be thought of as adding the meta data bonds and labels to a **cytnx.Tensor**. 
A **UniTensor** can be constructed with a **Bond** object, or directly from a **cytnx.Tensor** which gets converted to a **UniTensor**. 

In the following, let's look into these objects: 

.. toctree::
    :maxdepth: 3

    uniten/create.rst
    uniten/labels.rst
    uniten/bond.rst
    uniten/tagged.rst
    uniten/symmetric.rst
    uniten/blocks.rst
    uniten/elements.rst
    uniten/manipulation.rst