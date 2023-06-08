Accessing the block(s)
------------------------

In this section we introduce some basic ways to access and manipulate the blocks in UniTensors. This way, the data of a UniTensor can be accessed and manipulated. Each block is a **Tensor**.

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

Let's use the same example of UniTensor with U1 symmetry that we introduced in the previous section to demostrate how to get block(s) from a block structured UniTensor:

.. image:: image/u1_tdex.png
    :width: 500
    :align: center


.. code-block:: python
    :linenos:

    bond_c = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(1)>>1, cytnx.Qs(-1)>>1],[cytnx.Symmetry.U1()])
    bond_d = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(1)>>1, cytnx.Qs(-1)>>1],[cytnx.Symmetry.U1()])
    bond_e = cytnx.Bond(cytnx.BD_OUT, [cytnx.Qs(2)>>1, cytnx.Qs(0)>>2, cytnx.Qs(-2)>>1],[cytnx.Symmetry.U1()])
    Td = cytnx.UniTensor([bond_c, bond_d, bond_e])
    Td.set_name("Td")

There are two ways to get a certain block from a UniTensor. 

**1. Getting a block by its Qn-indices**

.. py:function:: UniTensor.get_block(qindices)

    :param List[int] qindices: list of integers specifying the indices of qnums on each bond

The quantum numbers need to be given in the same order as the legs in the tensor.

In our example we can access the block with qnums [Qs(1),Qs(-1),Qs(0)], which correspond to the Qn-indices [0,1,1]: 

.. code-block:: Python
    :linenos:

    B1 = Td.get_block_([0,1,1])
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

For example, if we want to get the block with block index number 1:

.. code-block:: python
    :linenos:

    B1 = Td.get_block_(1)
    print(B1)

Output >> 

.. code-block:: text

    Total elem: 2
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (1,1,2)
    [[[0.00000e+00 0.00000e+00 ]]]


.. note::

    The order of the blocks in a UniTensor depends on how the tensor was created. If you want to access blocks with certain quantum numbers, use *UniTensor.get_block(qindices)*.


**Getting all blocks:**

To access all valid blocks in a UniTensor with block structure (with symmetry), we can use **get_blocks()** or **get_blocks_()**. This will return the blocks as a *list* in Python or a *vector* in C++. Each block is a **cytnx.Tensor** object. The order of the blocks corresponds to their block indices.

* In Python:

.. code-block:: python
    :linenos:
    
    Blks = Td.get_blocks_()
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

**1. Putting a block into a location assigned by their Qn-indices**

.. py:function:: UniTensor.put_block(Tn, qindices)

    :param List[int] qindices: list of integers specifying the indices of qnums on each bond

We can, for example, put the block to the location in the UniTensor with qnums [Qs(1),Qs(-1),Qs(0)], corresponding to QN-indices [0,1,1]: 

* In Python:

.. code-block:: python
    :linenos:

    B2 = ones([1,1,2])
    B1 = Td.get_block_([0,1,1])
    print(B1)
    Td.put_block(B2,[0,1,1])
    print(Td.get_block_(1))

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

For example, if we want to put the tensor to the block with block index 1, then:

* In Python:

.. code-block:: python
    :linenos:

    B2 = ones([1,1,2])
    B1 = Td.get_block_(1)
    print(B1)
    Td.put_block(B2,1)
    print(Td.get_block_(1))
    


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
