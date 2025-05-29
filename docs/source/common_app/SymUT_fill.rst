Set same value for all blocks in UniTensor with Symmetry
-------------------------------------------------------------- 

Consider a UniTensor with Symmetry **Tsymm**. The following code provide example code to set all elements to be the same value (1.0 in this case) for all the blocks

* In Python:

     
.. code-block:: python
    :linenos:

    bond_d = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(1)>>1, cytnx.Qs(-1)>>1],[cytnx.Symmetry.U1()])
    bond_e = cytnx.Bond(cytnx.BD_IN, [cytnx.Qs(1)>>1, cytnx.Qs(-1)>>1],[cytnx.Symmetry.U1()])
    bond_f = cytnx.Bond(cytnx.BD_OUT,\
                        [cytnx.Qs(2)>>1, cytnx.Qs(0)>>2, cytnx.Qs(-2)>>1],[cytnx.Symmetry.U1()])
    Tsymm = cytnx.UniTensor([bond_d, bond_e, bond_f], name="symm. tensor", labels=["d","e","f"])

    for block in Tsymm.get_blocks_():
        block.fill(1.0)

    
* In C++:

.. code-block:: c++
    :linenos:

    for(auto &block: Td.get_blocks_()){
        block.fill(1.0);
    }

**Output (before) >>**

.. code-block:: text
    
    -------- start of print ---------
    Tensor name: Td
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
       [0] U1(1)   -->| 1     2 |-->  [1] U1(0)
                      |         |
       [1] U1(-1)  -->| 1       |
                      |         |
                      -----------

    Total elem: 2
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (1,1,2)
    [[[0.00000e+00 0.00000e+00 ]]]

    ========================
    BLOCK [#2]
     |- []   : Qn index 
     |- Sym(): Qnum of correspond symmetry
                      -----------
                      |         |
       [1] U1(-1)  -->| 1     2 |-->  [1] U1(0)
                      |         |
       [0] U1(1)   -->| 1       |
                      |         |
                      -----------

    Total elem: 2
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (1,1,2)
    [[[0.00000e+00 0.00000e+00 ]]]

    ========================
    BLOCK [#3]
     |- []   : Qn index 
     |- Sym(): Qnum of correspond symmetry
                      -----------
                      |         |
       [1] U1(-1)  -->| 1     1 |-->  [2] U1(-2)
                      |         |
       [1] U1(-1)  -->| 1       |
                      |         |
                      -----------

    Total elem: 1
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (1,1,1)
    [[[0.00000e+00 ]]]


**Output (after) >>**

.. code-block:: text

    -------- start of print ---------
    Tensor name: Td
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
    [[[1.00000e+00 ]]]

    ========================
    BLOCK [#1]
     |- []   : Qn index 
     |- Sym(): Qnum of correspond symmetry
                      -----------
                      |         |
       [0] U1(1)   -->| 1     2 |-->  [1] U1(0)
                      |         |
       [1] U1(-1)  -->| 1       |
                      |         |
                      -----------

    Total elem: 2
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (1,1,2)
    [[[1.00000e+00 1.00000e+00 ]]]

    ========================
    BLOCK [#2]
     |- []   : Qn index 
     |- Sym(): Qnum of correspond symmetry
                      -----------
                      |         |
       [1] U1(-1)  -->| 1     2 |-->  [1] U1(0)
                      |         |
       [0] U1(1)   -->| 1       |
                      |         |
                      -----------

    Total elem: 2
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (1,1,2)
    [[[1.00000e+00 1.00000e+00 ]]]

    ========================
    BLOCK [#3]
     |- []   : Qn index 
     |- Sym(): Qnum of correspond symmetry
                      -----------
                      |         |
       [1] U1(-1)  -->| 1     1 |-->  [2] U1(-2)
                      |         |
       [1] U1(-1)  -->| 1       |
                      |         |
                      -----------

    Total elem: 1
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (1,1,1)
    [[[1.00000e+00 ]]]




