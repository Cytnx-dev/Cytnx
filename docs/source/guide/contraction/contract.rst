Contract(s)
=============
Contractions of two tensors can be done with **Contract()**. Using this function, indices with the same labels on the two tensors are contracted. **Contracts()** provides the same functionality for more than two tensors. In this case, the contraction order can additionally be specified.

Contract
------------------

The function **cytnx.Contract()** contracts all common labels of two UniTensors. For example:

* In Python:

.. code-block:: python
    :linenos:


    A = cytnx.UniTensor(cytnx.ones([2,3,4]), rowrank = 1)
    A.relabels_(["i","j","l"])

    B = cytnx.UniTensor(cytnx.ones([3,2,4,5]), rowrank = 2)
    B.relabels_(["j","k","l","m"])

    C = cytnx.Contract(A, B)

    A.print_diagram()
    B.print_diagram()
    C.print_diagram()


Output >> 

.. code-block:: text

    -----------------------
    tensor Name : 
    tensor Rank : 3
    block_form  : False
    is_diag     : False
    on device   : cytnx device: CPU
              ---------     
             /         \    
       i ____| 2     3 |____ j
             |         |    
             |       4 |____ l
             \         /    
              ---------     
    -----------------------
    tensor Name : 
    tensor Rank : 4
    block_form  : False
    is_diag     : False
    on device   : cytnx device: CPU
              ---------     
             /         \    
       j ____| 3     4 |____ l
             |         |    
       k ____| 2     5 |____ m
             \         /    
              ---------     
    -----------------------
    tensor Name : 
    tensor Rank : 3
    block_form  : False
    is_diag     : False
    on device   : cytnx device: CPU
              ---------     
             /         \    
       i ____| 2     5 |____ m
             |         |    
       k ____| 2       |        
             \         /    
              ---------     

Here we see that the labels **j** and **l** appear on both input tensors. Thus, they are contracted. Note that the bond dimensions of the contracted tensors must agree on both tensors.

In order to define which indices shall be contracted without changing the labels on the initial tensors, Cyntx provides the method **.relabels()**. It allows to set common labels on the indices to be contracted and distinct labels on the others. Also, the labels on the resulting tensor can be defined this way. Suppose that we only want to contract the index *j* in the previous example, but not sum over *l*. We can use **.relabels()** for this task:


* In Python:

.. code-block:: python
    :linenos:


    A = cytnx.UniTensor(cytnx.ones([2,3,4]), rowrank = 1)
    A.relabels_(["i","j","l"])
    Are = A.relabels(["i","j","lA"])

    B = cytnx.UniTensor(cytnx.ones([3,2,4,5]), rowrank = 2)
    B.relabels_(["j","k","l","m"])
    Bre = B.relabels(["j","k","lB","m"])

    C = cytnx.Contract(Are, Bre)

    A.print_diagram()
    B.print_diagram()
    C.print_diagram()


Output >> 

.. code-block:: text

    -----------------------
    tensor Name : 
    tensor Rank : 3
    block_form  : False
    is_diag     : False
    on device   : cytnx device: CPU
              ---------     
             /         \    
       i ____| 2     3 |____ j
             |         |    
             |       4 |____ l
             \         /    
              ---------     
    -----------------------
    tensor Name : 
    tensor Rank : 4
    block_form  : False
    is_diag     : False
    on device   : cytnx device: CPU
              ---------     
             /         \    
       j ____| 3     4 |____ l
             |         |    
       k ____| 2     5 |____ m
             \         /    
              ---------     
        -----------------------
    tensor Name : 
    tensor Rank : 5
    block_form  : False
    is_diag     : False
    on device   : cytnx device: CPU
               ---------     
              /         \    
       i  ____| 2     2 |____ k
              |         |    
       lA ____| 4     4 |____ lB
              |         |    
              |       5 |____ m
              \         /    
               ---------     


The function **.relabels()** creates a copy of the initial UniTensor and changes the labels, while keeping the labels on the initial tensor unchanged. The actual data is shared between the old and new tensor, only the meta is independent.

Contracts
------------------
The function **Contracts** allows us to contract multiple UniTensors.

The first argument of this function is **TNs**, which is a list containing all UniTensors to be contracted. Contracts also provides the argument **order** to specify a desired contraction order, or the **optimal** option to use an auto-optimized contraction order.

Consider the following contraction task consisting of UniTensors **A1**, **A2** and **M**:

.. image:: image/contracts.png
    :width: 300
    :align: center

This corresponds to the Python program:

* In Python:

.. code-block:: python
    :linenos:

    
    # Creating A1, A2, M
    A1 = cytnx.UniTensor(cytnx.random.normal([2,8,8], mean=0., std=1., dtype=cytnx.Type.ComplexDouble), name = "A1");
    A2 = A1.Conj();
    A2.set_name("A2");
    M = cytnx.UniTensor(cytnx.ones([2,2,4,4]), name = "M")

    # Assign labels
    A1.relabels_(["phy1","v1","v2"])
    M.relabels_(["phy1","phy2","v3","v4"])
    A2.relabels_(["phy2","v5","v6"])

    # Use Contracts
    Res = cytnx.Contracts(TNs = [A1,M,A2], order = "(M,(A1,A2))", optimal = False)
    Res.print_diagram()


Output >> 

.. code-block:: text

    -----------------------
    tensor Name : 
    tensor Rank : 6
    block_form  : False
    is_diag     : False
    on device   : cytnx device: CPU
               ---------     
              /         \    
       v1 ____| 8     8 |____ v2
              |         |    
              |       4 |____ v3
              |         |    
              |       4 |____ v4
              |         |    
              |       8 |____ v5
              |         |    
              |       8 |____ v6
              \         /    
               ---------     

Note that the UniTensors' names have to be specified for an explicitly given contraction order. In this case we specified them in the constructor argument. The order *(M,(A1,A2))* indicates that first all common indices of *A1* and *A2* are contracted, then all common indices of the resulting tensor and *M*.

.. Note::
    All tensors contracted with `Contracts()` need to have unique tensor names. Use `UniTensor.set_name()` to specify the name of a tensor.