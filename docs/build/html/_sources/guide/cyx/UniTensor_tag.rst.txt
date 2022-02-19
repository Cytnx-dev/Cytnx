Tagged UniTensor 
-------------------

In this section we introduce the **tagged UniTensor**, which is the UniTensor with its bonds being *directional*,
in this case, depending on wheather the bonds have symmetriy (quantum number),
the resulting UniTensor can be further categorized into **non-symmetry** and with **symmetry** (block form).


Non-symmetry Tagged UniTensor
********************************

To create the non-symmetry Tagged UniTensor, we initialize it with **non-symmetry and direcitonal bonds**, although no symmetry can be exploit in these bonds,
The direction information still add some physical properties to the UniTensor.
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

Now we would like to emphasize that the direction informations add constrain on contracting two UniTensors, 
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


Symmetry Tagged UniTensor
********************************

Tensor with symmetries has several advantages in physical simulations, due to the constrain from the symmetries, we have less free parameter in the tensors
, both the memory needed and the cost of several tensor operations and algorithms can be substantially reduced.

To bring the symmetries into our UniTensor, we simply initialize it using bonds with **direction and the symmetry (quantum numbers)** properties. 
We can follow the 7.2. Bond section to create bonds **bond_c**, **bond_d** and **bond_e** with symmreties, and use them to initialize our UniTensor **Td**:

* In python:
  
.. code-block:: python
    :linenos:

    bond_c = cytnx.Bond(2, cytnx.BD_KET, [[1], [-1]])
    bond_d = cytnx.Bond(2, cytnx.BD_KET, [[1], [-1]])
    bond_e = cytnx.Bond(4, cytnx.BD_BRA, [[2], [0], [0], [-2]])
    Td = cytnx.UniTensor([bond_c, bond_d, bond_e], rowrank = 2)
    Td.set_name("Td")
    Td.print_diagram()


Output >> 

.. code-block:: text
 
    -----------------------
    tensor Name : Td
    tensor Rank : 3
    block_form  : true
    contiguous  : True
    valid bocks : 3
    is diag   : False
    on device   : cytnx device: CPU
    braket_form : True
            row               col 
            ---------------      
            |             |     
      0  -->| 2         3 |-->  2  
            |             |     
      1  -->| 2           |        
            |             |     
            ---------------   

We note that in this UniTensor, the **block_form** is **True**, and the **valid blocks** is **3**.

When the quantum numbers are given in the bonds, what Cytnx did when initilizing the UniTensor is combine all the bonds in the rowspace into one bond, and all the bonds in the colspace into the other bond,
results in a sparse tensor with 2 bonds, or, a block diagonalized matrix, each block in the matrix is labeled by a set of quantum numbers associate with the symmetries specified.

In our example above, two bonds in the rowspace are first combined together according to the U(1) combine rule, making it a single bond with quantum numbers 2,0 (with 2-fold degeneracy!) and -2,
now consider the only bond in the colspace and its quantum numbers, we expect that there will be 3 valid blocks labeled 2,0 and -2.

These blocks are what users can directly access in the symmetry tagged UniTensor, refer to the next section 7.3.3. Accessing the block(s).


.. image:: image/ut_bd.png
    :width: 600
    :align: center


.. image:: image/ut_blocks.png
    :width: 600
    :align: center



.. toctree::
