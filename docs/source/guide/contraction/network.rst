Network
========================

The **Network** object provides a skeleton of a Tensor network diagram. Users can write a *Network file*, which serves as the blue print of a TN structure. With a *Network* defined in such a way, multiple Tensors can be contracted at once. The contraction order can either be specified or automatically optimized by Cytnx. Furthermore, the tensor network can be printed to check the implementation.

*Network* is useful when we have to contract different tensors with the same connectivity many times. We can define the *Network* once and reuse it for several contractions. The contraction order can be optimized once after the first initialization with tensors. In proceeding steps this optimized order can be reused.

Network from .net file
--------------------------

We take a typical contraction in a corner transfer matrix algorithm as an example. The TN diagram is given by:

.. image:: image/ctm.png
    :width: 300
    :align: center

We implement the diagram as a .net file to represent the contraction task:

* ctm.net:

.. code-block:: text
    :linenos:

    c0: t0-c0, t3-c0
    c1: t1-c1, t0-c1
    c2: t2-c2, t1-c2
    c3: t3-c3, t2-c3
    t0: t0-c1, w-t0, t0-c0
    t1: t1-c2, w-t1, t1-c1
    t2: t2-c3, w-t2, t2-c2
    t3: t3-c0, w-t3, t3-c3
    w: w-t0, w-t1, w-t2, w-t3
    TOUT:
    ORDER: ((((((((c0,t0),c1),t3),w),t1),c3),t2),c2)

Note that:

1. The labels above correspond to the diagram you draw, not the label attribute of UniTensor objects. Both label conventions can, but do not have to be the same.
   
2. Labels should be separated by ' , '. In TOUT, a ' ; ' separates the labels in rowspace and colspace.
   
3. TOUT specifies the output configuration, in this case we leave it blank since the result will be a scalar.
   
4. ORDER is optional and used to specify the contraction order manually.

Put UniTensors and Launch
--------------------------
We use the .net file to create a Network. Then, we can load instances of UniTensors:

* In Python:

.. code-block:: python
    :linenos:

    # initialize tensors
    w = cytnx.UniTensor(cytnx.random.normal([2,2,2,2], mean=0., std=1.))
    c0 = cytnx.UniTensor(cytnx.random.normal([8,8], mean=0., std=1.))
    c1 = cytnx.UniTensor(cytnx.random.normal([8,8], mean=0., std=1.))
    c2 = cytnx.UniTensor(cytnx.random.normal([8,8], mean=0., std=1.))
    c3 = cytnx.UniTensor(cytnx.random.normal([8,8], mean=0., std=1.))
    t0 = cytnx.UniTensor(cytnx.random.normal([8,2,8], mean=0., std=1.))
    t1 = cytnx.UniTensor(cytnx.random.normal([8,2,8], mean=0., std=1.))
    t2 = cytnx.UniTensor(cytnx.random.normal([8,2,8], mean=0., std=1.))
    t3 = cytnx.UniTensor(cytnx.random.normal([8,2,8], mean=0., std=1.))

    # initialize network object from ctm.net file
    net = cytnx.Network("ctm.net")

    # put tensors
    net.PutUniTensor("w",w)  
    net.PutUniTensor("c0",c0)
    net.PutUniTensor("c1",c1)
    net.PutUniTensor("c2",c2)
    net.PutUniTensor("c3",c3)
    net.PutUniTensor("t0",t0)
    net.PutUniTensor("t1",t1)
    net.PutUniTensor("t2",t2)  
    net.PutUniTensor("t3",t3)

    print(net)

* In C++:

.. code-block:: c++
    :linenos:

    // initialize tensors
    w = cytnx.UniTensor(cytnx.random.normal({2,2,2,2}), 0., 1.);
    // and so on...

    // initialize network object from ctm.net file
    Network net = cytnx.Network("ctm.net");

    // put tensors
    net.PutUniTensor("c0", c0);
    net.PutUniTensor("t0", t0);
    net.PutUniTensor("c1", c1);
    // and so on...

    cout << net;

Output >> 

.. code-block:: text

    ==== Network ====
    [o] c0 : t0-c0 t3-c0 
    [o] c1 : t1-c1 t0-c1 
    [o] c2 : t2-c2 t1-c2 
    [o] c3 : t3-c3 t2-c3 
    [o] t0 : t0-c1 w-t0 t0-c0 
    [o] t1 : t1-c2 w-t1 t1-c1 
    [o] t2 : t2-c3 w-t2 t2-c2 
    [o] t3 : t3-c0 w-t3 t3-c3 
    [o] w : w-t0 w-t1 w-t2 w-t3 
    TOUT : ; 
    ORDER : ((((((((c0,t0),c1),t3),w),t1),c3),t2),c2)
    =================

To perform the contraction and get the outcome, we use the .Launch():

* In Python:

.. code-block:: python
    :linenos:

    Res = net.Launch(optimal = True)

* In C++:

.. code-block:: c++
    :linenos:

    UniTensor Res = net.Launch(true)

Here if the argument **optimal = True**, the contraction order is always auto-optimized.
If **optimal = False**, the specified ORDER in the .net file will be used. If ORDER is not specified, the order of the tensor definitions in the .net file is used.


.. Note::
    1. The auto-optimized contraction order obtained by calling **.Launch(optimal = True)** is saved in the Network object. If there is no need to re-optimize the order (i.e. the bond dimensions of the input tensors remain (approximately) the same.), we can put new tensors and call **.Launch()** again with **optimal = False**. In this case, the optimized order is reused, which avoids the overhead of recalculating the optimal order.
    2. The indices of the UniTensors to be put into the Network need to be ordered according to the indices in the .net file. Otherwise, the index order can be defined in PutTensor explicitly, see :ref:`PutUniTensor according to label ordering` below.

Network from string
--------------------------
Alternatively, we can implement the contraction directly in the program with FromString(): 

* In Python:

.. code-block:: python
    :linenos:

    net = cytnx.Network()
    net.FromString(["c0: t0-c0, t3-c0",\
                    "c1: t1-c1, t0-c1",\
                    "c2: t2-c2, t1-c2",\
                    "c3: t3-c3, t2-c3",\
                    "t0: t0-c1, w-t0, t0-c0",\
                    "t1: t1-c2, w-t1, t1-c1",\
                    "t2: t2-c3, w-t2, t2-c2",\
                    "t3: t3-c0, w-t3, t3-c3",\
                    "w: w-t0, w-t1, w-t2, w-t3",\
                    "TOUT:",\
                    "ORDER: ((((((((c0,t0),c1),t3),w),t1),c3),t2),c2)"])

This approach can be convenient if you do not want to maintain the .net files.

PutUniTensor according to label ordering
------------------------------------------

When we put a UniTensor into a Network, we can also specify its leg order by the bond labels in a UniTensor. This way, the user does not need to know or look up the order of the indices of the bonds. As an example, we consider two UniTensors **A** and **B** with three bonds each.  Both tensors have one leg corresponding to physical degrees of freedom and the other two legs are internal indices of the Tensor Network. Tensors of this kind are used in matrix product states, and the internal indices point to the left or right in diagrams, while the physical index is oriented vertically. We first create such tensors and set the corresponding labels:

* In Python:

.. code-block:: python
    :linenos:

    A = cytnx.UniTensor(cytnx.ones([2,8,8]));
    A.relabels_(["phy", "left", "right"])
    B = cytnx.UniTensor(cytnx.ones([2,8,8]));
    B.relabels_(["phy", "left", "right"])

The legs of these tensors are arranged such that the first leg is the physical leg (with dimension 2, corresponding to a spin-half chain) and the other two legs are
the internal bonds (with bond dimension 8).

If we want to contract the physical legs of the two tensors, we can create the following Network:

* In Python:

.. code-block:: python
    :linenos:

    net = cytnx.Network()
    net.FromString(["T0: v0in, phy, v0out",\
                    "T1: v1in, phy, v1out",\
                    "TOUT: v0in, v1in; v0out, v1out"])

Note that this Network uses the convention that the second legs of the tensors are contracted. This is not consistent with the index ordering of **A** and **B**, which have the physical leg in the first position. However, if we specify the labels when we put the tensors, we do not have to worry about the index order:


* In Python:

.. code-block:: python
    :linenos:

    net.PutUniTensor("T0", A, ["left", "phy", "right"])
    net.PutUniTensor("T1", B, ["left", "phy", "right"])
    Tout=net.Launch()
    Tout.print_diagram()

Output >> 

.. code-block:: text

    -----------------------
    tensor Name : 
    tensor Rank : 4
    block_form  : False
    is_diag     : False
    on device   : cytnx device: CPU
                 ---------     
                /         \    
       v0in ____| 8     8 |____ v0out
                |         |    
       v1in ____| 8     8 |____ v1out
                \         /    
                 ---------     

We added the bond labels as a third argument in PutUniTensor(). In this case, the indices will be permuted according to the label ordering of the Network.

Note that the names of tensors and indices can differ from the names and labels of the UniTensors, which makes it possible to flexibly reuse the Network for different tensor in consecutive contractions.


.. toctree::

