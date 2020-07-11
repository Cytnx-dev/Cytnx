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


Tensor Contraction 
------------------------
One of the most important operation of tensors is to multiply multiple tensors together, and summing over the indices. 
For example, consider three tensors :math:`A_{\alpha\beta\gamma}` a rank-3 tensor, :math:`B_{\beta\delta}` is a rank-2 tensor (matrix) and :math:`C_{\gamma}` is a rank-1 tensor (vector) multiply together to get a rank-2 tensor :math:`D_{\alpha\delta}` by summing over the common indices :math:`\beta,\gamma`. With einstain notation:

.. math::
    
    D_{\alpha\delta} = A_{\alpha\beta\gamma}B_{\beta\delta}C_{\gamma}


In the tensor notation, this is equivalent as the following diagram:

.. image:: image/cont.png
    :width: 550
    :align: center

where the dash line indicate connetion of two bonds with the same indices. Summing over a index simply reprent by the connection of bonds and two (or more) tensors are called to *contract* together. 


Direction of bond
-----------------------
Above, we have shown that every symbolic tensor notation can be represented by graphical tensor notation, and each bond can be interpreted as vector space. 
In quantum system, sometimes we want to indicate the physical space with *bra* (:math:`<\psi|`) and *ket* (:math:`|\psi>`). For example, a matrix can be generally represent as :math:`A = |\alpha><\beta|`, and *ket* can only multiply (contract) with a *bra*.  
In terms of tensor notation, we give each bond a **direction** (arrow) where bond with arrow pointing into the node indicates a *ket* and arrow pointing away from the node indicates a *bra* as shown in the following:

.. image:: image/braket.png
    :width: 500
    :align: center

(a) indicate a rank-3 tensor :math:`A = |\alpha><\beta|<\delta|` and (b) is a rank-2 tensor :math:`B = |\alpha><\beta|`. 
Just like in physics, a *ket* bond cannot multiply with a *bra* bond. In the tensor notation, this can be very straightforwardly presented, as two bonds with conflict direction cannot contract with each other, as shown in the following:

.. image:: image/braket_cont.png
    :width: 350
    :align: center


.. note:: 

    Generally, each bond can represent different basis that is not interchangable, especially in the more complicated cases where each bond carries different quantum number. 




.. toctree::

