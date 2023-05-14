Contraction
=============
Cytnx provides rich UniTenosr contraction interfaces, in this section we introcuce several methods to contract a desired tensor network.

Contract()
------------------

For the contraction of two UniTensor, we have the function **cytnx.Contract()** to do the job, what it does is simply contract 
the common labels of two UniTensors. Here is a example:

* In python:

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


* In python:

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

Contracts()
------------------
The function **Contracts** allow us to contract multiple Unitensors.

This function take an argument **TNs** which is a list contains UniTensors to be contracted,
we also provide arguments **order** for users to specify a desired contraction order, and an **optimal** option to use an auto-optimized contraction order.

Consider the following contraction task consists of UniTensors **A1**, **A2** and **M**:

.. image:: image/contracts.png
    :width: 300
    :align: center

translate to the code we have:

* In python:

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

ncon()
------------------
The **ncon** is another useful function to reduce users' programming effort required to implement a tesnor network contraction, which is orginally proposed for MATLAB :cite:`pfeifer2015ncon`.

To use ncon, we first make a labelled diagram of the desired network contraction such that:
Each internal index (index to be contracted) is labelled with a unique positive integer (typically sequential integers starting from 1, although this is not necessary).

External indices of the diagram (if there are any) are labelled with sequential negative integers [-1,-2,-3,…] which denote the desired index order on the final tensor (with -1 as the first index, -2 as the second etc).

Following this, the **ncon** routine is called as follows,

.. py:function:: OutputTensor = ncon(tensor_list_in, connect_list_in, cont_order)
     
    :param list tensor_list_in: 1D array containing the tensors comprising the network
    :param list connect_list_in: 1D array of vectors, where the kth element is a vector of the integer labels from the diagram on the kth tensor from tensor_list_in (ordered following the corresponding index order on this tensor).
    :param list cont_order: a vector containing the positive integer labels from the diagram, used to specify order in which **ncon** contracts the indices. Note that cont_order is an optional input that can be omitted if desired, in which case ncon will contract in ascending order of index lab.

For example, we want to contract the following tesnor network (again) consists of tensors **A1**, **A2** and **M**:

.. image:: image/ncon.png
    :width: 300
    :align: center

In the figure we labelled the internal leg using the unique positive numbers and extermal legs the negative ones, translate this figure
to the ncon function calling we have:

* In python:

.. code-block:: python
    :linenos:

    # Creating A1, A2, M
    A1 = cytnx.UniTensor(cytnx.ones([2,8,8]))
    A2 = cytnx.UniTensor(cytnx.ones([2,8,8]))
    M = cytnx.UniTensor(cytnx.ones([2,2,4,4]))

    # Calling ncon
    res = cytnx.ncon([A1,M,A2],[[1,-1,-2],[1,2,-3,-4],[2,-5,-6]])

We see that **ncon** accomplish the similar thing as **Contracts**, just now the labeling of the UniTensors in the network 
is incorporated into the function argument, thus make the code more compact.

.. bibliography:: ref.ncon.bib
    :cited:
