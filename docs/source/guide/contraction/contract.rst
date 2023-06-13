Contract(s)
=============
Cytnx provides rich UniTensor contraction interfaces, in this section we introduce several methods to contract a desired tensor network.

Contract
------------------

For the contraction of two UniTensor, we have the function **cytnx.Contract()** to do the job, what it does is simply contract 
the common labels of two UniTensors. Here is a example:

* In Python:

.. code-block:: python
    :linenos:


    A = cytnx.UniTensor(cytnx.ones([3,3,3]), rowrank = 1)
    A.set_labels([1,2,3])

    B = cytnx.UniTensor(cytnx.ones([3,3,3,3]), rowrank = 2)
    B.set_labels([2,3,4,5])

    C = cytnx.Contract(A, B)

    A.print_diagram()
    B.print_diagram()
    C.print_diagram()


Output >> 

.. code-block:: text

    -----------------------
    tensor Name : 
    tensor Rank : 3
    block_form  : false
    is_diag     : False
    on device   : cytnx device: CPU
             -------------      
            /             \     
      1 ____| 3         3 |____ 2  
            |             |     
            |           3 |____ 3  
            \             /     
             -------------      
    -----------------------
    tensor Name : 
    tensor Rank : 4
    block_form  : false
    is_diag     : False
    on device   : cytnx device: CPU
             -------------      
            /             \     
      2 ____| 3         3 |____ 4  
            |             |     
      3 ____| 3         3 |____ 5  
            \             /     
             -------------     
    -----------------------
    tensor Name : 
    tensor Rank : 3
    block_form  : false
    is_diag     : False
    on device   : cytnx device: CPU
             -------------      
            /             \     
      1 ____| 3         3 |____ 4  
            |             |     
            |           3 |____ 5  
            \             /     
             -------------   

Here we see that legs from two UniTensors with same labels **2** and **3** are contracted.

Often we have to change the labels of UniTensors in order to do the desired contraction, to reduce the inconvinence of maintaining
labels, we provide **.relabels()** to relabel the UniTensors, this in fact return us a copy of UniTensor to do the contraction job,
while the labels of the original UniTensor itself are preserved. Here is the example:


* In Python:

.. code-block:: python
    :linenos:


    A = cytnx.UniTensor(cytnx.ones([3,3,3]), rowrank = 1)
    A.set_labels([1,2,3])
    At = A.relabels([-1,-2,-3])

    B = cytnx.UniTensor(cytnx.ones([3,3,3]), rowrank = 1)
    B.set_labels([4,5,6])
    Bt = B.relabels([-3,-4,-5])

    C = cytnx.Contract(At, Bt)

    A.print_diagram()
    B.print_diagram()
    C.print_diagram()


Output >> 

.. code-block:: text

    -----------------------
    tensor Name : 
    tensor Rank : 3
    block_form  : false
    is_diag     : False
    on device   : cytnx device: CPU
             -------------      
            /             \     
      1 ____| 3         3 |____ 2  
            |             |     
            |           3 |____ 3  
            \             /     
             -------------      
    -----------------------
    tensor Name : 
    tensor Rank : 3
    block_form  : false
    is_diag     : False
    on device   : cytnx device: CPU
             -------------      
            /             \     
      3 ____| 3         3 |____ 4  
            |             |     
            |           3 |____ 5  
            \             /     
             -------------      
    -----------------------
    tensor Name : 
    tensor Rank : 4
    block_form  : false
    is_diag     : False
    on device   : cytnx device: CPU
             -------------      
            /             \     
      1 ____| 3         3 |____ 2  
            |             |     
            |           3 |____ 4  
            |             |     
            |           3 |____ 5  
            \             /     
             -------------   

Note that in this example, two UniTensors **A** and **B** have no labels in common, but we somehow want to contract them while
preserving their labels, that's the reason why we use **.relabels** here.

Contracts
------------------
The function **Contracts** allow us to contract multiple Unitensors.

This function take an argument **TNs** which is a list contains UniTensors to be contracted,
we also provide arguments **order** for users to specify a desired contraction order, and an **optimal** option to use an auto-optimized contraction order.

Consider the following contraction task consists of UniTensors **A1**, **A2** and **M**:

.. image:: image/contracts.png
    :width: 300
    :align: center

translate to the code we have:

* In Python:

.. code-block:: python
    :linenos:

    # Creating A1, A2, M
    A1 = cytnx.UniTensor(cytnx.ones([2,8,8]), name = "A1")
    A2 = cytnx.UniTensor(cytnx.ones([2,8,8]), name = "A2")
    M = cytnx.UniTensor(cytnx.ones([2,2,4,4]), name = "M")

    # Assign labels
    A1.set_labels(["phy1","v1","v2"])
    M.set_labels(["phy1","phy2","v3","v4"])
    A2.set_labels(["phy2","v5","v6"])

    # Use Contracts
    res = cytnx.Contracts(TNs = [A1,M,A2], order = "(M,(A1,A2))", optimal = False)

Note that to specify the contraction orders, the UniTensors' name should be specified(in this case we specified them in the constructor argument).