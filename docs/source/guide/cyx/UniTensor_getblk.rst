Accessing the block(s)
------------------------

In this section we introduce some basic ways to access and manipulate the blocks in UniTensors.

UniTensor without symmetries
*****************************

For the UniTensor without symmetries, we expect the UniTensor is just an Tensor with bonds labeled by ids,
in this case, the **.get_block()**  or **.get_block()_** will return the Tensor object of the UniTensor for us to manipulate.

* In python:

.. code-block:: python
    :linenos:

    # Create an UniTensor from Tensor
    T = cytnx.UniTensor(cytnx.ones([3,3]), rowrank=1)
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

As an example for symmetry UniTensor, we consider the two-sites local Hamitonian **H** appeared in iTEBD algorithm when U(1) symmetry is exploited.
We create the U(1) symmetric **H** and use **.get_blocks_qnums()** to check the infomation of its blocks (associate qunatum number labels).

* In python:

.. code-block:: python
    :linenos:

    bdi = cytnx.Bond(2,cytnx.BD_KET,[[1],[-1]])
    bdo = bdi.clone().set_type(cytnx.BD_BRA)
    H = cytnx.UniTensor([bdi,bdi,bdo,bdo], labels=[2,3,0,1], rowrank=2)
    print(H.get_blocks_qnums())


Output >> 

.. code-block:: text

    Vector Print:
    Total Elements:3
    [Vector Print:
    Total Elements:1
    [-2]
    , Vector Print:
    Total Elements:1
    [0]
    , Vector Print:
    Total Elements:1
    [2]
    ]

In this case we have **3 blocks**, labeled by **3 set of quantum numbers [-2], [0], [2]** (which are also represented as vectors), since only U(1)
symmetry are used, each label contains only 1 quantum number.

We can further use **.get_blocks_()** or **.get_blocks()** to get all the blocks in a vector/list of an symmetric UniTensor


* In python:

.. code-block:: python
    :linenos:

    print(H.get_blocks_())


Output >> 

.. code-block:: text

    Total elem: 1
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (1,1)
    [[0.00000e+00 ]]



    Total elem: 4
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (2,2)
    [[0.00000e+00 0.00000e+00 ]
    [0.00000e+00 0.00000e+00 ]]



    Total elem: 1
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (1,1)
    [[0.00000e+00 ]]


    [, , ]

To get a block with desired quantum numbers, we use **.get_block_()** or **.get_block()**, with the argument to be the quantum number label.
Note again that the blocks are nothing but the normal **Tensor** object, so we can manipulate it as we did to the Tensors,
here we demostrate the usage of **.get_block_()**, since the return should be the reference, we can directly assign/modify the content of these blocks.

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
