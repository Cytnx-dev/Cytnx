Tagged UniTensor 
-------------------

In this section we introduce **tagged UniTensor**, which is a UniTensor with *directional* bonds. Mathematically, a bond directing towards a tensor represents to a vectorspace, while a bond directing away from a tensor represents the corresponding dual space. In physics, this is often represented as a "Ket" or "Bra" vector. We use the convention that "Ket" is represented by a bond directing towards the tensor, while "Bra" points away from the tensor.

If symmetric tensors are considered, the bonds carry quantum numbers. Therefore, UniTensor can further be categorized into **non-symmetric** and **symmetric** (block form).

Non-symmetric Tagged UniTensor
********************************

To create the non-symmetric, tagged UniTensor, we initialize it with **directional** bonds (BD_IN/BD_KET or BD_OUT/BD_BRA). Let's create UniTensors **Ta**, **Tb** and **Tc** in this way and investigate their properties:

* In Python:
  
.. code-block:: python
    :linenos:

    bond_a = cytnx.Bond(3, cytnx.BD_KET)
    bond_b = cytnx.Bond(3, cytnx.BD_BRA)
    Ta = cytnx.UniTensor([bond_a, bond_a, bond_b], labels=[0, 1, 2], rowrank = 2)
    Ta.set_name("Ta")
    Ta.print_diagram()
    Tb = cytnx.UniTensor([bond_a, bond_b, bond_b], labels=[2, 3, 4], rowrank = 1)
    Tb.set_name("Tb")
    Tb.print_diagram()
    Tc = cytnx.UniTensor([bond_b, bond_b, bond_b], labels=[2, 3, 4], rowrank = 1)
    Tc.set_name("Tc")
    Tc.print_diagram()


Output >> 

.. code-block:: text
    
    -----------------------
    tensor Name : Ta
    tensor Rank : 3
    block_form  : false
    is_diag     : False
    on device   : cytnx device: CPU
    braket_form : True
            row               col   
            ---------------      
            |             |     
      0  -->| 3         3 |-->  2  
            |             |     
      1  -->| 3           |        
            |             |     
            ---------------     
    -----------------------
    tensor Name : Tb
    tensor Rank : 3
    block_form  : false
    is_diag     : False
    on device   : cytnx device: CPU
    braket_form : True
            row               col   
            ---------------      
            |             |     
      2  -->| 3         3 |-->  3  
            |             |     
            |           3 |-->  4  
            |             |     
            ---------------      
    -----------------------
    tensor Name : Tc
    tensor Rank : 3
    block_form  : false
    is_diag     : False
    on device   : cytnx device: CPU
    braket_form : False
            row               col   
            ---------------      
            |             |     
      2 *<--| 3         3 |-->  3  
            |             |     
            |           3 |-->  4  
            |             |     
            ---------------     
            
In this example, the UniTensors **Ta** and  **Tb**  are created to be in the **braket form (braket_form : True)**. This means that all bonds in the rowspace are Kets (inward) and all the bonds in the colspace are Bras(outward), which can be seen in the diagram as well. The property rowrank of a UniTensor defines these spaces: the first bonds are part of rowspace, and rowspace contains rowrank bonds.

Two bonds can only be contracted when one of them corresponds to a Ket state (living in a vector space) and the other is Bra state (living in the dual space). Therefore, **two bonds with conflicting directions cannot be contract with each other**. While this is certainly a constraint, this condition makes the implementation of tensor network algorithms less error-prone since incompatible bond directions will not be contracted.

To demonstrate this, we can just try:


* In Python:
  
.. code-block:: python
    :linenos:

    cytnx.Contract(Ta, Tb)
    cytnx.Contract(Ta, Tc)

While the first contraction works well, the second one throws a RuntimeError stating "BRA-KET mismatch":


Output >> 

.. code-block:: text
    
    RuntimeError: 
    # Cytnx error occur at virtual boost::intrusive_ptr<cytnx::UniTensor_base> cytnx::DenseUniTensor::contract(const boost::intrusive_ptr<cytnx::UniTensor_base>&, const bool&, const bool&)
    # error: [ERROR][DenseUniTensor][contract] cannot contract common label: <2> @ self bond#2 & rhs bond#0, BRA-KET mismatch!




.. toctree::
