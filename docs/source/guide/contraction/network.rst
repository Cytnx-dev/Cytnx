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

1. The labels above correspond to the diagram you draw, not the label attribute of UniTensor objects. Both label conventions can but do not have to be the same.
   
2. Labels should be seperated by ' , '. A ' ; ' seperates the labels in rowspace and colspace. In the above case all bonds are in the colspace.
   
3. TOUT specifies the output configuration, in this case we leave it blank since the result will be a scalar.
   
4. ORDER is optional and used to specify the contraction order manually.

Put UniTensors and Launch
--------------------------
We use the .net file to create a Network. Then, we can load instances of UniTensors:

* In Python:

.. code-block:: python
    :linenos:

    N = Network("ctm.net")
    N.PutUniTensor("c0",c0)
    N.PutUniTensor("t0",t0)
    print(N)
    N.PutUniTensor("c1",c1)
    # and so on...

* In C++:

.. code-block:: c++
    :linenos:

    Network N = Network("ctm.net");
    N.PutUniTensor("c0",c0);
    N.PutUniTensor("t0",t0);
    cout << N;
    N.PutUniTensor("c1",c1)
    // and so on...

Output >> 

.. code-block:: text

    ==== Network ====
    [o] c0 : t0-c0 t3-c0 
    [x] c1 : t1-c1 t0-c1 
    [x] c2 : t2-c2 t1-c2 
    [x] c3 : t3-c3 t2-c3 
    [o] t0 : t0-c1 w-t0 t0-c0 
    [x] t1 : t1-c2 w-t1 t1-c1 
    [x] t2 : t2-c3 w-t2 t2-c2 
    [x] t3 : t3-c0 w-t3 t3-c3 
    [x] w : w-t0 w-t1 w-t2 w-t3 
    TOUT : ; 
    ORDER : ((((((((c0,t0),c1),t3),w),t1),c3),t2),c2)
    =================
To perform the contraction and get the outcome, we use the Launch():

* In Python:

.. code-block:: python
    :linenos:

    Res = N.Launch(optimal = True)

* In C++:

.. code-block:: c++
    :linenos:

    UniTensor Res = N.Launch(true)

Here if the argument **optimal = True**, the contraction order is always auto-optimized.
If **optimal = False**, the specified ORDER in the .net file will be used. If ORDER is not specified, the order of the tensor definitions in the .net file is used.


.. Note::
    The auto-optimized contraction order obtained by calling **.Launch(optimal = True)** will save in the Network object with the optimized order. If there is no need to re-optimize the order (i.e. the order of the bond dimensions of the input tensors remain the same.), we can put new tensors and call **.Launch()** again with **optimal = False**. In this case, the optimized order is reused, which avoids the overhead of recalculating the optimal order.

Network from string
--------------------------
Alternatively, we can implement the contraction directly in the program with FromString(): 

* In Python:

.. code-block:: python
    :linenos:

    N = cytnx.Network()
    N.FromString(["c0: t0-c0, t3-c0",\
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

This approach should be convenient when you do not want to maintain the .net file outside the program.

PutUniTensor according to label ordering
------------------------------------------

When we put a UniTensor into a Network, we can also specify its leg order by the bond labels in a UniTensor. This way, the user does not need to know or look up the order of the indices of the bonds. As an example, we consider two UniTensors **A1** and **A2** with three bonds each.  Both tensors have one leg corresponding to physical degrees of freedom and the other two legs are internal indices of the Tensor Network. Tensors of this kind are used in matrix product states, and the internal indices point to the left or right in diagrams, while the physical index is oriented vertically. We first create such tensors and set the corresponding labels:

* In Python:

.. code-block:: python
    :linenos:

    T0 = cytnx.UniTensor(cytnx.ones([2,8,8]));
    T0.set_labels(["phy","left","right"])
    T1 = cytnx.UniTensor(cytnx.ones([2,8,8]));
    T1.set_labels(["phy","left","right"])

The legs of these tensors are arranged such that the first leg is the physical leg (with dimension 2, corresponding to a spin-half chain) and the other two legs are
the internal bonds (with bond dimension 8).

If we want to contract the physical legs of the two tensors, we can create the following Network:

* In Python:

.. code-block:: python
    :linenos:

    N = cytnx.Network()
    N.FromString(["A0: v0in, phy0, v0out",\
                "A1: v1in, phy1, v1out",\
                "TOUT: v0in, v1in, v0out, v1out"])

Note that this Network uses the convention that the second legs of the tensors are contracted. This is not consistent with the index ordering of **T0** and **T1**, which have the physical leg in the first position. However, if we specify the labels when we put the tensors, we do not have to worry about the index order:


* In Python:

.. code-block:: python
    :linenos:

    N.PutUniTensor("A0",T0,["left","phy","right"])
    N.PutUniTensor("A1",T1,["left","phy","right"])

We added the bond labels as a third argument in PutUniTensor(). In this case, the indices will be permuted according to the label ordering of the Network.

Note that the names of tensors and indices can differ from the names and labels of the UniTensors, which makes it possible to flexibly reuse the Network for different tensor in consecutive contractions.


.. toctree::

