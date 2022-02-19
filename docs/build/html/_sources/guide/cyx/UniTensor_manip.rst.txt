Manipulate UniTensor
--------------------

Having introduced the detail of initialization and structure of three types (un-tagged, tagged and tagged with symmetry) of UniTensor,
in this section we present several general and useful hints on dealing with UniTensors.


Permutation, Reshape, Arithmetic ...
************************************

Basic operations like permutation, reshape and arithmetic for un-tagged type UniTensor can be done exacly the same as Tensors,
refer to the section 2.2. Manipulate Tensor and 2.4. Tensor arithmetic.

For the manipulations on elements, we just remember to first access to the blocks (which are just tensors objects) and then access the elements as
we have learned in 2.3. Access elements.

UniTensor contraction
*****************************

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

For the contraction task on multiple Unitensors, refer to 7.4. Network.

Rowrank
*********

Another property that we may want to maintain in UniTensor is its rowrank, it simply tells us
how the legs of the a UniTensor is splited into two halves, namely one part belongs to the rowspace
and the other the column space. Most of the linear algebra algorithms takes the matrix as the input, we thus use the rowrank
to specify how to cast the input UniTensor into a matrix. In Cytnx this specification will sometimes reduce the amounts of
work (in the code) to do linalg operations on UniTensor, here is a example about performing **SVD decomposition** on a UniTensor.


* In python:

.. code-block:: python
    :linenos:

    T = cytnx.UniTensor(cytnx.ones([5,5,5,5,5]), rowrank = 3)
    s, u, vt = cytnx.linalg.Svd(T)

    T.print_diagram()
    s.print_diagram()
    u.print_diagram()
    vt.print_diagram()


Output >> 

.. code-block:: text

    -----------------------
    tensor Name : 
    tensor Rank : 5
    block_form  : false
    is_diag     : False
    on device   : cytnx device: CPU
             -------------      
            /             \     
      0 ____| 5         5 |____ 3  
            |             |     
      1 ____| 5         5 |____ 4  
            |             |     
      2 ____| 5           |        
            \             /     
             -------------      
    -----------------------
    tensor Name : 
    tensor Rank : 2
    block_form  : false
    is_diag     : True
    on device   : cytnx device: CPU
             -------------      
            /             \     
     -1 ____| 25       25 |____ -2 
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
      0 ____| 5        25 |____ -1 
            |             |     
      1 ____| 5           |        
            |             |     
      2 ____| 5           |        
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
     -2 ____| 25        5 |____ 3  
            |             |     
            |           5 |____ 4  
            \             /     
             -------------    

.. toctree::

Here we note that when doing SVD, the first three legs of **T** are actually automatically reshape into one leg according to **rowrank=3**, 
after the task, the **u** again automatically be reshaped back to three legs in the rowspace, so when we try to contract back **u-s-vt** we expect to 
get the original **T** with same shape.