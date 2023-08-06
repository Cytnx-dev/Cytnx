Manipulate UniTensor
--------------------

After having introduced the initialization and structure of the three UniTensor types (un-tagged, tagged and tagged with symmetries),
we show the basic functionalities to manipulate UniTensors.

Permutation, reshaping and arithmetic operations are accessed similarly to **Tensor** objects as introduced before, with slight modifications for symmetric UniTensors.

Permute
************************************

The bond order can be changed with *permute* for all kinds of UniTensors. The order can either be defined by the index order as for the permute method of a *Tensor*, or by specifying the label order after the permutation.

For example, we permute the indices of the symmetric tensor that we introduced before:

* In Python:

.. code-block:: python
    :linenos:

    bond_d = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(1)>>1, cytnx.Qs(-1)>>1],[cytnx.Symmetry.U1()])
    bond_e = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(1)>>1, cytnx.Qs(-1)>>1],[cytnx.Symmetry.U1()])
    bond_f = cytnx.Bond(cytnx.BD_OUT,\
                        [cytnx.Qs(2)>>1, cytnx.Qs(0)>>2, cytnx.Qs(-2)>>1],[cytnx.Symmetry.U1()])
    Tsymm = cytnx.UniTensor([bond_d, bond_e, bond_f], name="symm. tensor").relabels_(["d","e","f"])
    Tsymm.print_diagram()

    Tsymm_perm_ind=Tsymm.permute([2,0,1])
    Tsymm_perm_ind.print_diagram()

    Tsymm_perm_label=Tsymm.permute(["f","d","e"])
    Tsymm_perm_label.print_diagram()


* Output >> 

.. code-block:: text

      -----------------------
      tensor Name : symm. tensor
      tensor Rank : 3
      contiguous  : True
      valid blocks : 4
      is diag   : False
      on device   : cytnx device: CPU
            row           col 
               -----------    
               |         |    
         d  -->| 2     4 |-->  f
               |         |    
         e  -->| 2       |        
               |         |    
               -----------    

      -----------------------
      tensor Name : symm. tensor
      tensor Rank : 3
      contiguous  : False
      valid blocks : 4
      is diag   : False
      on device   : cytnx device: CPU
            row           col 
               -----------    
               |         |    
         f *<--| 4     2 |<--* e
               |         |    
         d  -->| 2       |        
               |         |    
               -----------    

      -----------------------
      tensor Name : symm. tensor
      tensor Rank : 3
      contiguous  : False
      valid blocks : 4
      is diag   : False
      on device   : cytnx device: CPU
            row           col 
               -----------    
               |         |    
         f *<--| 4     2 |<--* e
               |         |    
         d  -->| 2       |        
               |         |    
               -----------   


We did the same permutation in two ways in this example, once using indices, once using labels. The first index of the permuted tensor corresponds to the last index of the original tensor (original index 2, label "f"), the second new index to the first old index (old index 0, label "d") and the last new bond has the old index 1 and label "e".

Reshape
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

Combine bonds
************************************

The tagged UniTensors include symmetric UniTensors cannot be reshaped, since the bonds to be combinded or splited now includes the direction and quantum number infomation,
the reshape process involves the fusion or split of the qunatum basis, we provide combindBonds API for the tagged UniTensor as an alternative to the usual reshape function.
Note that currently there is no API for splitting a bond, since the way to split the quantum basis will be ambiguous.
Let's see the complete function usage for combining bonds:


.. py:function:: UniTensor.combineBonds(indicators, force)	
     
    :param list indicators: A list of **integer** indicating the indices of bonds to be combined. If a list of **string** is passed the bonds with those string labels will be combined.
    :param bool force: If set to **True** the bonds will be combined regardless the direction or type of the bonds, otherwise the bond types will be checked. The default is **False**.


Consider a specific example:

* In Python:

.. code-block:: python
    :linenos:

      from cytnx import Bond, BD_IN, BD_OUT, Qs, Symmetry
      # bond1 = Bond(BD_IN,[[2,0], [4,1]],[3,5],[Symmetry.U1(), Symmetry.Zn(2)])
      # bond2 = Bond(BD_IN,[Qs(2,0)>>3, Qs(4,1)>>5],[Symmetry.U1(), Symmetry.Zn(2)])
      bd1 = cytnx.Bond(cytnx.BD_IN,[[1],[-1]],[1,1])
      bd2 = cytnx.Bond(cytnx.BD_IN,[[1],[-1]],[1,1])
      bd3 = cytnx.Bond(cytnx.BD_OUT,[[2],[0],[0],[-2]],[1,1,1,1])

      ut = cytnx.UniTensor([bd1,bd2,bd3],rowrank=2)
      print(ut)
      
      ut.combineBonds([0,1])
      print(ut)

Output >> 

.. code-block:: text

      -------- start of print ---------
      Tensor name: 
      braket_form : True
      is_diag    : False
      [OVERALL] contiguous : True
      ========================
      BLOCK [#0]
      |- []   : Qn index 
      |- Sym(): Qnum of correspond symmetry
                    -----------
                    |         |
      [0] U1(1)  -->| 1     1 |-->  [0] U1(2)
                    |         |
      [0] U1(1)  -->| 1       |
                    |         |
                    -----------

      Total elem: 1
      type  : Double (Float64)
      cytnx device: CPU
      Shape : (1,1,1)
      [[[0.00000e+00 ]]]

      ========================
      BLOCK [#1]
      |- []   : Qn index 
      |- Sym(): Qnum of correspond symmetry
                     -----------
                     |         |
      [0] U1(1)   -->| 1     1 |-->  [1] U1(0)
                     |         |
      [1] U1(-1)  -->| 1       |
                     |         |
                     -----------

      Total elem: 1
      type  : Double (Float64)
      cytnx device: CPU
      Shape : (1,1,1)
      [[[0.00000e+00 ]]]

      ========================
      BLOCK [#2]
      |- []   : Qn index 
      |- Sym(): Qnum of correspond symmetry
                     -----------
                     |         |
      [0] U1(1)   -->| 1     1 |-->  [2] U1(0)
                     |         |
      [1] U1(-1)  -->| 1       |
                     |         |
                     -----------

      Total elem: 1
      type  : Double (Float64)
      cytnx device: CPU
      Shape : (1,1,1)
      [[[0.00000e+00 ]]]

      ========================
      BLOCK [#3]
      |- []   : Qn index 
      |- Sym(): Qnum of correspond symmetry
                     -----------
                     |         |
      [1] U1(-1)  -->| 1     1 |-->  [1] U1(0)
                     |         |
      [0] U1(1)   -->| 1       |
                     |         |
                     -----------

      Total elem: 1
      type  : Double (Float64)
      cytnx device: CPU
      Shape : (1,1,1)
      [[[0.00000e+00 ]]]

      ========================
      BLOCK [#4]
      |- []   : Qn index 
      |- Sym(): Qnum of correspond symmetry
                     -----------
                     |         |
      [1] U1(-1)  -->| 1     1 |-->  [2] U1(0)
                     |         |
      [0] U1(1)   -->| 1       |
                     |         |
                     -----------

      Total elem: 1
      type  : Double (Float64)
      cytnx device: CPU
      Shape : (1,1,1)
      [[[0.00000e+00 ]]]

      ========================
      BLOCK [#5]
      |- []   : Qn index 
      |- Sym(): Qnum of correspond symmetry
                     -----------
                     |         |
      [1] U1(-1)  -->| 1     1 |-->  [3] U1(-2)
                     |         |
      [1] U1(-1)  -->| 1       |
                     |         |
                     -----------

      Total elem: 1
      type  : Double (Float64)
      cytnx device: CPU
      Shape : (1,1,1)
      [[[0.00000e+00 ]]]




      # Cytnx warning occur at void cytnx::Bond_impl::combineBond_(const boost::intrusive_ptr<cytnx::Bond_impl>&, const bool&)
      # warning: [WARNING] duplicated qnums might appears!

      # file : /home/j9263178/Cytnx/src/Bond.cpp (327)
      -------- start of print ---------
      Tensor name: 
      braket_form : False
      is_diag    : False
      [OVERALL] contiguous : True
      ========================
      BLOCK [#0]
      |- []   : Qn index 
      |- Sym(): Qnum of correspond symmetry
                    ----------
                    |        |
      [2] U1(2)  -->| 1      |
                    |        |
      [2] U1(2) *<--| 1      |
                    |        |
                    ----------

      Total elem: 1
      type  : Double (Float64)
      cytnx device: CPU
      Shape : (1,1)
      [[0.00000e+00 ]]

      ========================
      BLOCK [#1]
      |- []   : Qn index 
      |- Sym(): Qnum of correspond symmetry
                    ----------
                    |        |
      [1] U1(0)  -->| 2      |
                    |        |
      [1] U1(0) *<--| 2      |
                    |        |
                    ----------

      Total elem: 4
      type  : Double (Float64)
      cytnx device: CPU
      Shape : (2,2)
      [[0.00000e+00 0.00000e+00 ]
      [0.00000e+00 0.00000e+00 ]]

      ========================
      BLOCK [#2]
      |- []   : Qn index 
      |- Sym(): Qnum of correspond symmetry
                     ----------
                     |        |
      [0] U1(-2)  -->| 1      |
                     |        |
      [0] U1(-2) *<--| 1      |
                     |        |
                     ----------

      Total elem: 1
      type  : Double (Float64)
      cytnx device: CPU
      Shape : (1,1)
      [[0.00000e+00 ]]
            


Arithmetic
************************************


Arithmetic operations for un-tagged UniTensors can be done exactly the same as with Tensors, see :ref:`Tensor arithmetic`. The supported arithmetic operations and further linear algebra functions are listed in :ref:`Linear algebra`.

Rowrank
*********

Another property that we may want to maintain in UniTensor is its rowrank. It tells us how the legs of the a UniTensor are split into two halves, one part belongs to the rowspace and the other to the column space. A UniTensor can then be seen as a linear operator between these two spaces, or as a matrix. The matrix results in having the first *rowrank* indices combined to the first (row-)index and the other indices combined to the second (column-)index. Most of the linear algebra algorithms take a matrix as an input. We thus use rowrank to specify how to cast the input UniTensor into a matrix. In Cytnx, this specification makes it easy to use linear algebra operations on UniTensors. Here is an example where a **singular value decomposition (SVD)** is performed on a UniTensor:


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

When calling *Svd*, the first three legs of **T** are automatically reshaped into one leg according to **rowrank=3**. After the SVD, the matrices **U** and **Vt** are automatically reshaped back into the corresponding index form of the original tensor. This way, we get the original UniTensor **T** if we contract :math:`U \cdot S \cdot Vt`:

* In Python:

.. code-block:: python
    :linenos:

    Tsymm_diff=T-cytnx.Contracts([U,S,Vt]);
    Tsymm_diff.Norm()/T.Norm()

Output >> 

.. code-block:: text

      -----------------------
      Total elem: 1
      type  : Double (Float64)
      cytnx device: CPU
      Shape : (1)
      [7.87947e-16 ]

If we contract :math:`U \cdot S \cdot Vt`, we get a tensor of the same shape as **T** and we can subtract the two tensors. The error :math:`\frac{|T-U \cdot S \cdot Vt|}{|T|}` is of the order of machine precision, as expected.