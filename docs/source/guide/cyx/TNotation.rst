Tensor Notation
==================
Before going into the objects in the cytnx_extension, let's first introduce the useful notations people usually use in the tensor network.

A tensor can be think of as a multi-dimensional array. For example, a rank-0 tensor is scalar, rank-1 tensor is vector and rank-2 tensor is matrix etc. 
Mathmatically, a rank-N tensor can be written in symbolic notation as 

.. math::
    
    T_{i_1,i_2 \cdots, i_N}

This notation is standard, but when considering lots of tensors multiply together, the expression is sometimes quite difficult to read and not easy to understand. 
This is where the graphical tensor notation is useful. 

Usually, in the standard tensor network paper, the tensor notation is quite common and frequently used to explain the tensor network algorithms. Let's take a look at it. 

Each tensor constitude a node (sometimes it also called vertex in graph theory) and several bonds (legs) attach to it. 
The number of bonds represent the *rank* of a tensor. 


.. image:: image/Not.png
    :width: 600
    :align: center

For example as show in above, (a) represent a rank-0 tensor, which is a scalar. (b) is a rank-1 tensor which is a *vector*, (c) is a rank-2 tensor which is *matrix* and (d) is a rank-3 tensor. 


Tensor dot, Contraction 
------------------------



.. toctree::

