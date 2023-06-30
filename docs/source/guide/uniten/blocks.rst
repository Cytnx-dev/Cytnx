Accessing the block(s)
------------------------

The data in a UniTensor is stored in blocks. We introduce how to access and manipulate these. This way, the data of a UniTensor can be accessed and manipulated. Each block is a **Tensor**.

UniTensor without symmetries
*****************************

A UniTensor without symmetries is simply a Tensor with labeled bonds. In this case, the methods **.get_block()**  and **.get_block_()** return the Tensor object of the UniTensor for us to manipulate.

* In Python:

.. code-block:: python
    :linenos:

    # Create an UniTensor from Tensor
    T = cytnx.UniTensor(cytnx.ones([3,3]))
    print(T.get_block())

Output >> 

.. code-block:: text

    Total elem: 9
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (3,3)
    [[1.00000e+00 1.00000e+00 1.00000e+00 ]
    [1.00000e+00 1.00000e+00 1.00000e+00 ]
    [1.00000e+00 1.00000e+00 1.00000e+00 ]]



.. note::

    While **.get_block_()** returns a reference to the Tensor which corresponds to the data in a UniTensor, **.get_block()** makes a copy of the data. Therefore, changes to a Tensor returned from **.get_block_()** will also change the UniTesor data, while changes after a **.get_block()** do not affect the UniTensor.


UniTensor with symmetries
*****************************

Let's use the same example of a UniTensor with *U1* symmetry that we introduced in the previous section :ref:`UniTensor with Symmetries` to demonstrate how to get block(s) from a block structured UniTensor:

.. image:: image/u1_tdex.png
    :width: 500
    :align: center

* In Python:

.. code-block:: python
    :linenos:

    bond_d = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(1)>>1, cytnx.Qs(-1)>>1],[cytnx.Symmetry.U1()])
    bond_e = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(1)>>1, cytnx.Qs(-1)>>1],[cytnx.Symmetry.U1()])
    bond_f = cytnx.Bond(cytnx.BD_OUT,\
                        [cytnx.Qs(2)>>1, cytnx.Qs(0)>>2, cytnx.Qs(-2)>>1],[cytnx.Symmetry.U1()])
    Tsymm = cytnx.UniTensor([bond_d, bond_e, bond_f], name="symm. tensor").relabels_(["d","e","f"])

There are two ways to get a certain block from a UniTensor. 

**1. Getting a block by its quantum number indices**

.. py:function:: UniTensor.get_block(qindices)

    :param List[int] qindices: list of integers specifying the indices of the quantum numbers on each bond

The quantum number indices (*qindices*) need to be given in the same order as the legs in the tensor. In our example, *bond\_f* was created with three quantum numbers. Their indices are

    * 0 for *U1(2)*
    * 1 for *U1(0)*
    * 2 for *U1(-2)*
because the quantum numbers were created in this order. Similarly for *bond\_d* and *bond\_e: *U1(1)* has quantum number index 0 and *U1(-1)* has quantum number index 2.


As an example, we want to access the block with quantum numbers [Qs(1),Qs(-1),Qs(0)]. In the above convention, this corresponds to the quantum number indices [0,1,1]: 

* In Python:

.. code-block:: Python
    :linenos:

    B1 = Tsymm.get_block_([0,1,1])
    print(B1)

Output >> 

.. code-block:: text

    Total elem: 2
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (1,1,2)
    [[[0.00000e+00 0.00000e+00 ]]]


**2. Getting a block by its block index**
   
.. py:function:: UniTensor.get_block(blockindex)

    :param [int] blockindex: the index of the block in the UniTensor

If we know the block index, we can access the data directly. For example, if we want to get the block with block index number 1:

* In Python:

.. code-block:: python
    :linenos:

    B1 = Tsymm.get_block_(1)
    print(B1)

Output >> 

.. code-block:: text

    Total elem: 2
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (1,1,2)
    [[[0.00000e+00 0.00000e+00 ]]]


.. note::

    The order of the blocks in a UniTensor depends on how the tensor was created. If you want to access blocks with certain quantum number indices, use *UniTensor.get_block(qindices)*.


**Getting all blocks:**

To access all valid blocks in a UniTensor with block structure (with symmetries), we can use **get_blocks()** or **get_blocks_()**. This will return the blocks as a *list* in Python or a *vector* in C++. Each block is a **cytnx.Tensor** object. The order of the blocks corresponds to their block indices.

* In Python:

.. code-block:: python
    :linenos:
    
    Blks = Tsymm.get_blocks_()
    print(len(Blks))
    print(*Blks)

Output >> 

.. code-block:: text
    
    4

    Total elem: 1
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (1,1,1)
    [[[0.00000e+00 ]]]



    Total elem: 2
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (1,1,2)
    [[[0.00000e+00 0.00000e+00 ]]]



    Total elem: 2
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (1,1,2)
    [[[0.00000e+00 0.00000e+00 ]]]



    Total elem: 1
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (1,1,1)
    [[[0.00000e+00 ]]]


.. note::

    Note again that **.get_block(s)_()** returns to the block in the UniTensor, while **.get_block(s)()** creates an independent copy of the data.


Putting a block
************************

We might want to do some manipulations to an individual block that we got from *.get_block(s)*. Then, the new block can be put back to the UniTensor with *put_block()*.

**1. Putting a block into a location assigned by their quantum number indices**

.. py:function:: UniTensor.put_block(Tn, qindices)

    :param List[int] qindices: list of integers specifying the indices of quantum numbers on each bond

We can, for example, put the block to the location in the UniTensor with quantum numbers [Qs(1),Qs(-1),Qs(0)], corresponding to quantum number indices [0,1,1]: 

* In Python:

.. code-block:: python
    :linenos:

    B1new = cytnx.ones([1,1,2])
    B1 = Tsymm.get_block_([0,1,1])
    print(B1)
    Tsymm.put_block(B1new,[0,1,1])
    print(Tsymm.get_block_([0,1,1]))

Output >> 

.. code-block:: text

    Total elem: 2
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (1,1,2)
    [[[0.00000e+00 0.00000e+00 ]]]

    Total elem: 2
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (1,1,2)
    [[[1.00000e+00 1.00000e+00 ]]]


.. toctree::


**2. Putting a block by its block index**
   
.. py:function:: UniTensor.put_block(Tn, blockindex)

    :param [int] blockindex: the index of the blocks in the UniTensor

For example, if we want to put the tensor to the block with block index 2, then:

* In Python:

.. code-block:: python
    :linenos:

    B2new = cytnx.ones([1,1,2])
    B2 = Tsymm.get_block_(2)
    print(B2)
    Tsymm.put_block(B2new,2)
    print(Tsymm.get_block_(2))
    


.. code-block:: text

    Total elem: 2
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (1,1,2)
    [[[0.00000e+00 0.00000e+00 ]]]


    Total elem: 2
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (1,1,2)
    [[[1.00000e+00 1.00000e+00 ]]]
