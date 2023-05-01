Network
==========
Network is a class for contracting UniTensors, it is useful when we have to perform the same large contraction task many times.
We can create configuration for the contraction task and put the "constant" UniTensors in at the initialization step of some algorithms, 
later when runing the sweeping or the iterative steps, we put the variational tensors in and launch the network to get the results.

Network from .net file
************************
Let's take the corner transfer matrix for example, first we draw the desired tensor network diagram:

.. image:: image/ctm.png
    :width: 300
    :align: center

We now convert the diagram to the .net file to represent the contraction task, which is straghtforward:

* ctm.net:

.. code-block:: text
    :linenos:

    c1: ;0,2
    t1: ;1,3,0
    c2: ;4,1
    t2: ;9,6,4
    c3: ;11, 9
    t3: ;10, 8, 11
    c4: ;7, 10
    t4: ;2,5,7
    w: ;3,6,8,5
    TOUT:
    ORDER: ((((((((c1,t1),c2),t4),w),t2),c4),t3),c3)

Note that:

1. The labels above correspond to the diagram you draw, not the label attribute of UniTensor object itself.
   
2. Labels should be seperated by ' , ', and ' ; ' seperate the labels in rowspace and colspace. In the above case all legs live in the colspace.
   
3. TOUT specify the output configuration, in this case we leave it blank since we will get a scalar outcome.
   
4. ORDER is optional and used to specify the contraction order manually.

Put UniTensors and Launch
**************************
To use, we simply create the Network object (at the same time we load the .net file), and put the UniTensors:

* In python:

.. code-block:: python
    :linenos:

    N = Network("ctm.net")
    N.PutUniTensor("c1",c1)
    N.PutUniTensor("c2",c2)
    print(N)
    N.PutUniTensor("c3",c3)
    # and so on...

* In C++:

.. code-block:: c++
    :linenos:

    Network N = Network("ctm.net");
    N.PutUniTensor("c1",c1);
    N.PutUniTensor("c2",c2);
    cout << N;
    N.PutUniTensor("c3",c3)
    // and so on...

Output >> 

.. code-block:: text

    ==== Network ====
    [o] c1 : ; 0 2 
    [x] t1 : ; 1 3 0 
    [o] c2 : ; 4 1 
    [x] t2 : ; 9 6 4 
    [x] c3 : ; 11 9 
    [x] t3 : ; 10 8 11 
    [x] c4 : ; 7 10 
    [x] t4 : ; 2 5 7 
    [x] w : ; 3 6 8 5 
    TOUT : ; 
    ORDER : ((((((((c1,t1),c2),t4),w),t2),c4),t3),c3)
    =================

To perform the contraction and get the outcome, we use the Launch():

* In python:

.. code-block:: python
    :linenos:

    Res = N.Launch(optimal = True)

* In C++:

.. code-block:: c++
    :linenos:

    UniTensor Res = N.Launch(true)

Note that if the argument optimal = True, the contraction ORDER is always auto-optimized.
If optimal = False, the specified ORDER in network file will be used, otherwise contract one by one in sequence.

Network from string
********************
Alternatively, we can implement the contraction directly in the program with FromString(): 

* In python:

.. code-block:: python
    :linenos:

    N = cytnx.Network()
    N.FromString(["c1: ;0, 2",\
                "t1: ;1, 3, 0",\
                "c2: ;4, 1",\
                "t2: ;9, 6, 4",\
                "c3: ;11, 9",\
                "t3: ;10, 8, 11",\
                "c4: ;7, 10",\
                "t4: ;2, 5, 7",\
                "w: ;3, 6, 8, 5",\
                "TOUT:",\
                "ORDER: ((((((((c1,t1),c2),t4),w),t2),c4),t3),c3)"])

This approach should be convenient when you don't want to maintain the .net file outside the program.


PutUniTensor according to label ordering
*******************************************

When we put a UniTensor into a Network, we can also specify its leg order according to a label ordering, this interface turns out to be convinient
since users don't need to memorize or look up the index of s desired leg. To be more specific, consider
a example, we grab two three leg tensors **A1** and **A2**, they both have one leg that spans the physical space and the other two legs describe
the virtual space (such tensors are often appearing as the building block tensors of matrix product state), we create the tensors and set the corresponding lebels
as following:

* In python:

.. code-block:: python
    :linenos:

    A1 = cytnx.UniTensor(cytnx.ones([2,8,8]));
    A1.set_labels(["phy","v1","v2"])
    A2 = cytnx.UniTensor(cytnx.ones([2,8,8]));
    A2.set_labels(["phy","v1","v2"])

The legs of these tensors are arranged such that the first leg is the physical leg (with dimension 2 for spin-half case for example) and the other two legs are
the virtual ones (with dimension 8).

Now suppose somehow we want to contract these two tensors by its physical legs, we create the following Network:

* In python:

.. code-block:: python
    :linenos:

    N = cytnx.Network()
    N.FromString(["A1: 1,-1,2",\
                "A2: 3,-1,4",\
                "TOUT: 1,3;2,4"])

Note that in this Network it is the second leg of the two tensors to be contracted, which will not be consistent since **A1** and **A2**
are created such that their physical leg is the first one, while we can do the following:


* In python:

.. code-block:: python
    :linenos:

    N.PutUniTensor("A1",A1,["v1","phy","v2"])
    N.PutUniTensor("A2",A2,["v1","phy","v2"])

So when we do the PutUniTensor() we add the third arguement which is a labels ordering, what this function will do is nothing but permute
the tensor legs according to this label ordering before putting them into the Network.  


.. toctree::

