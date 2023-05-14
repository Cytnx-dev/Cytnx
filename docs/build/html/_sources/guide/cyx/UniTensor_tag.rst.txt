Tagged UniTensor 
-------------------

In this section we introduce the **tagged UniTensor**, which is the UniTensor with its bonds being *directional*. Physically, a bond with direction point into or outward a Tensor can be interpreted as a "Ket" or "Bra" physical state 

In this case, depending on wheather the bonds have symmetriy (quantum number),
the resulting UniTensor can be further categorized into **non-symmetry** and with **symmetry** (block form).


Non-symmetry Tagged UniTensor
********************************

To create the non-symmetry Tagged UniTensor, we initialize it with **direcitonal** (BD_IN/BD_KET or BD_OUT/BD_BRA) explicitly specified.
As mentioned above, the direction information add physical meaning to the UniTensor, as they explicitly specify each bond space being ket-space or bra-space.  
Let's create UniTensors **Ta**, **Tb** and **Tc** in this way and investigate their properties:

* In python:
  
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
            
We note that in this example, UniTensors **Ta** and  **Tb**  are created to be in the **bracket form (braket_form : True)**,
the condition for a UniTensor to be in bracket form is that, all the bonds in the rowspace are Kets (inward) and all the bonds in the colspace are Bras(outward),
which can be checked easily from the diagram.

As what we learned from quantum mechanic 101, two bonds can only be contract when one of them are Ket state and the other is Bra state. This leads to additional constrain on contracting two UniTensors, 
as we mentioned in the 7.1 Tensor notation section, **two bonds with conflict direction cannot contract with each other**.

To demostrate this, we can just try:


* In python:
  
.. code-block:: python
    :linenos:

    cytnx.Contract(Ta, Tb)
    cytnx.Contract(Ta, Tc)

We will find that the first one works well, while the second one throws a RuntimeError complaining "BRA-KET mismatch":


Output >> 

.. code-block:: text
    
    RuntimeError: 
    # Cytnx error occur at virtual boost::intrusive_ptr<cytnx::UniTensor_base> cytnx::DenseUniTensor::contract(const boost::intrusive_ptr<cytnx::UniTensor_base>&, const bool&, const bool&)
    # error: [ERROR][DenseUniTensor][contract] cannot contract common label: <2> @ self bond#2 & rhs bond#0, BRA-KET mismatch!




.. toctree::
