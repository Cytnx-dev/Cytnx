linalg extension
==================

.. .. toctree::
..     :maxdepth: 3

Tensor decomposition
**************************

As mention in the **Manipulate UniTensor**, the specification of **rowrank** makes it convinient to apply linear algebra operations on UniTensors. Here is an example where a **singular value decomposition (SVD)** is performed on a UniTensor:


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