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


.. Note:: 
    
    get_block_() with underscore indicate a reference view of that block! 

    Note again that the blocks are nothing but the normal **Tensor** object, so we can manipulate it as we did to the Tensors, here we demostrate the usage of **.get_block_()**, since the return should be the reference, we can directly assign/modify the content of these blocks.

[NEED MODIFY!!]

* In python:

.. code-block:: python
    :linenos:

    ## assign:
    # Q = 2  # Q = 0:    # Q = -2:
    # [1/4]    [[ -1/4, 1/2]     [1/4]
    #         [  1/2, -1/4]]

    H.get_block_([2])[0] = 1/4;
    T0 = H.get_block_([0])
    T0[0,0] = T0[1,1] = -1/4;
    T0[0,1] = T0[1,0] = 1/2;
    H.get_block_([-2])[0] = 1/4;

Alternatively, the above can also be done by the **.put_block_()** or **.put_block()**, with the argument to be a tensor and a quantum number label to indicate
which block to put in (replaced by the input tensor).

* In python:

.. code-block:: python
    :linenos:

    ## assign:
    # Q = 2  # Q = 0:    # Q = -2:
    # [1/4]    [[ -1/4, 1/2]     [1/4]
    #         [  1/2, -1/4]]

    Ta = cytnx.zeros([1,1])
    Ta[0,0] = 1/4
    Tb = cytnx.zeros([2,2])
    Tb[0,0] = Tb[1,1] = -1/4
    Tb[0,1] = Tb[1,0] = 1/2
    Tc = cytnx.zeros([1,1])
    Tc[0,0] = 1/4

    H.put_block_(Ta, [2])
    H.put_block_(Tb, [0])
    H.put_block_(Tc, [-2])

    print(H.get_blocks())


Output >> 

.. code-block:: text

    Total elem: 1
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (1,1)
    [[2.50000e-01 ]]



    Total elem: 4
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (2,2)
    [[-2.50000e-01 5.00000e-01 ]
    [5.00000e-01 -2.50000e-01 ]]



    Total elem: 1
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (1,1)
    [[2.50000e-01 ]]


    [, , ]



.. toctree::
