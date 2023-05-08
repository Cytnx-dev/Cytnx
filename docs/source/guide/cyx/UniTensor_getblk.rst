Accessing the block(s)
------------------------

In this section we introduce some basic ways to access and manipulate the blocks in UniTensors.

UniTensor without symmetries
*****************************

For the UniTensor without symmetries, we expect the UniTensor is just an Tensor with bonds labeled by ids,
in this case, the **.get_block()**  or **.get_block_()** will return the Tensor object of the UniTensor for us to manipulate.

* In python:

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

UniTensor with symmetries
*****************************

**Getting all blocks:**


Let's use the same example of U1 symmetry UniTensor we introduce earlier in previous section to demostrate how to get block(s) from a block structured UniTensor:

.. image:: image/u1_tdex.png
    :width: 500
    :align: center


.. code-block:: python
    :linenos:

    bond_c = cytnx.Bond(cytnx.BD_IN, [Qs(1)>>1, Qs(-1)>>1],[cytnx.Symmetry.U1()])
    bond_d = cytnx.Bond(cytnx.BD_IN, [Qs(1)>>1, Qs(-1)>>1],[cytnx.Symmetry.U1()])
    bond_e = cytnx.Bond(cytnx.BD_OUT, [Qs(2)>>1, Qs(0)>>2, Qs(-2)>>1],[cytnx.Symmetry.U1()])
    Td = cytnx.UniTensor([bond_c, bond_d, bond_e])
    Td.set_name("Td")

To access all the valid blocks from a UniTensor with block structure ( with symmetry), one can use **get_blocks()** or **get_blocks_()**. This will return a *list* (or *vector* in C++) of blocks (where each block is a **cytnx.Tensor** object) In the current order stored in UniTensor.


* In python:

.. code-block:: python
    :linenos:
    
    Blks = Td.get_blocks_()
    print(len(Blks))
    print(Blks)

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


    [, , , ]



.. Note::

    Again, just like other functions, with underscore means that its inplace, so shared view (reference) will be return. Without underscore the return will be the *copy* of all blocks! 


Getting a given block
**********************************

There are two ways to get a certain block from a UniTensor. 

1. getting a block by their index in the current order
   
.. py:function:: UniTensor.get_block(index)

    :param [int] index: the index of the blocks in current UniTensor

For example, if we want to get the 1-th block, then:

.. code-block:: python
    :linenos:

    B1 = Td.get_block_(1)
    print(B1)


.. code-block:: text

    Total elem: 2
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (1,1,2)
    [[[0.00000e+00 0.00000e+00 ]]]


2. getting a block by their Qn-indices

.. py:function:: UniTensor.get_block(qindices)

    :param List[int] qindices: a list of integer specify the indices of qnums on each axis/leg. 

    Alternatively, to get the 1-th block with qnums [Qs(1),Qs(-1),Qs(0)] on each leg, which corresponding to the Qn-indices [0,1,1], then: 

.. code-block:: python
    :linenos:

    B1 = Td.get_block_([0,1,1])
    print(B1)


.. code-block:: text

    Total elem: 2
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (1,1,2)
    [[[0.00000e+00 0.00000e+00 ]]]


.. Note:: 
    
    get_block_() with underscore indicate a reference view of that block! 

    Note again that the blocks are nothing but the normal **Tensor** object, so we can manipulate it as we did to the Tensors, here we demostrate the usage of **.get_block_()**, since the return should be the reference, we can directly assign/modify the content of these blocks.



Putting a given block
************************

Sometimes, the application might require user to put data into a given symmetry block. To do this, one can make use of *put_block()*

1. put a block by their index in the current order
   
.. py:function:: UniTensor.put_block(Tn, index)

    :param [int] index: the index of the blocks in current UniTensor

For example, if we want to put the tensor into the 1-th block location, then:

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



2. puting a block into location assigned by their Qn-indices

.. py:function:: UniTensor.put_block(Tn, qindices)

    :param List[int] qindices: a list of integer specify the indices of qnums on each axis/leg. 

    Alternatively, to put the tensor into the 1-th block with qnums [Qs(1),Qs(-1),Qs(0)] on each leg, which corresponding to the Qn-indices [0,1,1], then: 

.. code-block:: python
    :linenos:

    B2 = ones([1,1,2])
    B1 = Td.get_block_([0,1,1])
    print(B1)
    Td.put_block(B2,[0,1,1])
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


.. toctree::
