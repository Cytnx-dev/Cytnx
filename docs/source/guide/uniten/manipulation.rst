Manipulate UniTensor
--------------------

After having introduced the initialization and structure of the three UniTensor types (un-tagged, tagged and tagged with symmetries),
we show the basic functionalities to manipulate UniTensors.

Permutation, reshaping and arithmetic operations are accessed similarly to **Tensor** objects as introduced before, with slight modifications for symmetric UniTensors.

permute:
************************************

The bond order can be changed with *permute* for all kinds of UniTensors. For example, we permute the indices of the symmetric tensor that we introduced before:

* In Python:

.. code-block:: python
    :linenos:

      bond_c = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(1)>>1, cytnx.Qs(-1)>>1],[cytnx.Symmetry.U1()])
      bond_d = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(1)>>1, cytnx.Qs(-1)>>1],[cytnx.Symmetry.U1()])
      bond_e = cytnx.Bond(cytnx.BD_OUT, [cytnx.Qs(2)>>1, cytnx.Qs(0)>>2, cytnx.Qs(-2)>>1],[cytnx.Symmetry.U1()])
      Td = cytnx.UniTensor([bond_c, bond_d, bond_e]);
      Td.print_diagram()

      Td_perm=Td.permute([0,2,1])
      Td_perm.print_diagram()


* Output >> 

.. code-block:: text

      -----------------------
      tensor Name : 
      tensor Rank : 3
      contiguous  : True
      valid blocks : 4
      is diag   : False
      on device   : cytnx device: CPU
            row           col 
            -----------    
            |         |    
      0  -->| 2     4 |-->  2
            |         |    
      1  -->| 2       |        
            |         |    
            -----------    

      -----------------------
      tensor Name : 
      tensor Rank : 3
      contiguous  : False
      valid blocks : 4
      is diag   : False
      on device   : cytnx device: CPU
            row           col 
            -----------    
            |         |    
      0  -->| 2     2 |<--* 1
            |         |    
      2 *<--| 4       |        
            |         |    
            -----------    


reshape:
************************************

Untagged UniTensors can be reshaped just like normal Tensors.

* In Python:

.. code-block:: python
    :linenos:

    T = cytnx.UniTensor(cytnx.arange(12).reshape(4,3))
    T.reshape_(2,3,2)
    T.print_diagram()


Output >> 

.. code-block:: text

      -----------------------
      tensor Name : 
      tensor Rank : 3
      block_form  : False
      is_diag     : False
      on device   : cytnx device: CPU
            --------     
            /        \    
            |      2 |____ 0
            |        |    
            |      3 |____ 1
            |        |    
            |      2 |____ 2
            \        /    
            --------     

.. Note::

    A tagged UniTensor can not be reshaped. This includes symmetric UniTensors as well.


Arithmetic
************************************


Arithmetic operations for un-tagged UniTensors can be done exactly the same as with Tensors, see section arithmetic for Tensors.

rowrank
*********

Another property that we may want to maintain in UniTensor is its rowrank. It tells us how the legs of the a UniTensor are split into two halves, one part belongs to the rowspace and the other to the column space. A UniTensor can then be seen as a linear operator between these two spaces, or as a matrix. The matrix is be seen as having the first *rowrank* indices combined to the first (row-)index and the other indices combined to the second (column-)index. Most of the linear algebra algorithms take a matrix as an input. We thus use rowrank to specify how to cast the input UniTensor into a matrix. In Cytnx, this specification makes it easy to use linalg operations on UniTensors. Here is an example where a **singular value decomposition (SVD)** is performed on a UniTensor:


* In Python:

.. code-block:: python
    :linenos:

    T = cytnx.UniTensor(cytnx.ones([5,5,5,5,5]), rowrank = 3)
    S, U, Vt = cytnx.linalg.Svd(T)
    U.set_name('U')
    S.set_name('S')
    Vt.set_name('Vt')


    T.print_diagram()
    S.print_diagram()
    U.print_diagram()
    Vt.print_diagram()


Output >> 

.. code-block:: text

    -----------------------
    tensor Name : 
    tensor Rank : 5
    block_form  : False
    is_diag     : False
    on device   : cytnx device: CPU
              ---------     
             /         \    
       0 ____| 5     5 |____ 3
             |         |    
       1 ____| 5     5 |____ 4
             |         |    
       2 ____| 5       |        
             \         /    
              ---------     
    -----------------------
    tensor Name : S
    tensor Rank : 2
    block_form  : False
    is_diag     : True
    on device   : cytnx device: CPU
                   -----------     
                  /           \    
       _aux_L ____| 25     25 |____ _aux_R
                  \           /    
                   -----------     
    -----------------------
    tensor Name : U
    tensor Rank : 4
    block_form  : False
    is_diag     : False
    on device   : cytnx device: CPU
              ----------     
             /          \    
       0 ____| 5     25 |____ _aux_L
             |          |    
       1 ____| 5        |        
             |          |    
       2 ____| 5        |        
             \          /    
              ----------     
    -----------------------
    tensor Name : Vt
    tensor Rank : 3
    block_form  : False
    is_diag     : False
    on device   : cytnx device: CPU
                   ----------     
                  /          \    
       _aux_R ____| 25     5 |____ 3
                  |          |    
                  |        5 |____ 4
                  \          /    
                   ----------     

.. toctree::

When calling *Svd*, the first three legs of **T** are automatically reshaped into one leg according to **rowrank=3**. After the SVD, the matrices **U** and **Vt** are automatically reshaped back into the corresponding index form of the original tensor. This way, we get the original **T** if we contract :math:`U \cdot S \cdot Vt`:

* In Python:

.. code-block:: python
    :linenos:

    Tdiff=T-cytnx.Contracts([u,s,vt]);
    Tdiff.Norm()/T.Norm()

Output >> 

.. code-block:: text

      -----------------------
      Total elem: 1
      type  : Double (Float64)
      cytnx device: CPU
      Shape : (1)
      [7.87947e-16 ]

If we contract :math:`U \cdot S \cdot Vt`, we get a tensor of the same shape as **T** and we can subtract the two tensors. The error :math:`\frac{|T-U \cdot S \cdot Vt|}{|T|}` is of the order of machine precision.