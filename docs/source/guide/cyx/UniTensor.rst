UniTensor
==========
UniTensor is an extension object base on **cytnx.Tensor**. 
A UniTensor consist of three imporant parts: 

.. image:: image/UT.png
    :width: 600
    :align: center


1. Block(s), which is a **Cytnx.Tensor** (or a list/vector of **cytnx.Tensor**). Block(s) store all the numbers(elements). 

2. Bonds, which defines the bond information associate to each rank. 

3. labels, which gives each rank/bond unique ID. 



In the most simple example where we don't consider any symmetry in our system, UniTensor can be think of adding these bonds and labels meta data around **cytnx.Tensor**. 
A UniTensor can be constructed using **Bond** object, or in the simple case directly converted from a **Cytnx.Tensor**. 

Let's take a look at this:

.. toctree::
    :maxdepth: 1

    UniTensor_create.rst
    UniTensor_tag.rst
    UniTensor_sym.rst
    UniTensor_getblk.rst
    UniTensor_manip.rst
