CyTensor
==========
CyTensor is an extension object base on **cytnx.Tensor**. 
A CyTensor consist of three imporant parts: 1. Block, which is a **Cytnx.Tensor**, 2. Bonds, which defines the bond information associate to each rank. 3. labels, which gives each rank/bond unique ID. In the most simple example where we don't consider any symmetry in our system, CyTensor can be think of adding these bonds and labels meta data around **cytnx.Tensor**. 

A CyTensor can be constructed using **Bond** object, or in the simple case directly converted from a **Cytnx.Tensor**. 

Let's take a look at this:

.. toctree::
    :maxdepth: 1

    CyTensor_1_create.rst
    CyTensor_2_manip.rst
