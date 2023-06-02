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