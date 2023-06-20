UniTensor with Symmetry
---------------------------

Physical systems are often symmetric under certain transformations. Exploiting such symmetries can be advantageous in many cases. Cytnx allows to incorporate the symmetries on the level of the tensors directly.

In a system with a symmetry, charge is conserved. Therefore, the Hamiltonian can be block-diagonalized into symmetry sectors of defined charge (quantum numbers) -- the charge sectors do not mix. When the symmetric structure is imposed on the level of the tensors, they become block-diagonal as well. This substantially reduces the memory costs and the number of variations parameters in variational algorithms. Thus, larger system sizes or bond dimensions are accessible with the same computational costs. Moreover, numerical errors due to contributions that do not conserve symmetry and are therefore unphysical can be avoided. Finally, simulations can be restricted to certain symmetry sectors if needed.  

A quantum number conserving tensor can be understood in a simple way. Each bond (leg) of the Tensor carries quantum numbers and is directional, as shown in the following figure:

.. image:: image/ut_bd.png
    :width: 600
    :align: center

The conservation of charge (quantum numbers) is ensured by restricting the tensors to only have elements with **zero flux**. For these elements, the total quantum number flowing into the tensor has to be equal to the total quantum number flowing out. The combination of quantum numbers in incoming and outgoing links to a total quantum number is defined by the combine rule of the symmetry considered. All other elements, that do not conserve the charge, belong to an invalid block and therefore have to be zero. We do not store these invalid blocks explicitly in a UniTensor.


.. image:: image/ut_blocks.png
    :width: 600
    :align: center


To impose symmetry on the tensors, there are only two steps we need to do:

1. Identify the symmetries in the system (for example, U(1) symmetry)
2. Create **directional** Bonds that carry quantum numbers associated to these symmetries

As a simple example, lets create a 3-rank tensor with U(1) symmetry:

.. image:: image/u1_tdex.png
    :width: 500
    :align: center

Here, we use the notation *{Qnum}>>dimension*. First, three bonds  **bond_c** (in), **bond_d** (in) and **bond_e** (out) are created with corresponding qnums. See chapter Bond for further information related to the bonds and their creation with quantum numbers. We then initialize a UniTensor **Td** using these three bonds:

* In Python:
  
.. code-block:: python
    :linenos:

    bond_c = cytnx.Bond(cytnx.BD_IN, [Qs(1)>>1, Qs(-1)>>1],[cytnx.Symmetry.U1()])
    bond_d = cytnx.Bond(cytnx.BD_IN, [Qs(1)>>1, Qs(-1)>>1],[cytnx.Symmetry.U1()])
    bond_e = cytnx.Bond(cytnx.BD_OUT, [Qs(2)>>1, Qs(0)>>2, Qs(-2)>>1],[cytnx.Symmetry.U1()])
    Td = cytnx.UniTensor([bond_c, bond_d, bond_e])
    Td.set_name("Td")
    Td.print_diagram()


Output >> 

.. code-block:: text
 
    -----------------------
    tensor Name : Td
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

As shown in the previous figure, this UniTensor has only **4** valid blocks which carry zero-flux. We can use **Td.print_blocks()** to see the number of blocks and their structures:

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


.. Note::

    The number in the square braket **[]** in print_blocks() indicates the Qnum index. It refers to the order of the quantum numbers on the corresponding bond. In the previous example, *bond_e* contains three quantum numbers. The element with quantum number U1(2) has the Qnum index [0] on this link, the U1(0) elements have Qnum index [1], and U1(2) element has Qnum index [2]. More information on the output can be found in :ref:`print() and print_blocks()`.






.. toctree::
