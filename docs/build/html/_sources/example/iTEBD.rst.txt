iTEBD
------------
Time evolution block decimation is one of the most simple and sucessful Tensor network method :cite:`itebd-vidal`. The core concept of this algorithm is to use the imaginary time evolution to find the best variational ansatz, usually in terms of Matrix product state (MPS). 


Here, we use a 1D transverse field Ising model (TFIM) as a simple example to show how to implement iTEBD algorithm in Cytnx and get the infinite system size (variational) ground state. 

Consider the Hamiltonain of TFIM:

.. math::

    H = J\sum_{ij} \sigma^{z}_i\sigma^{z}_j - H_x\sum_i \sigma^{x}_i

where :math:`\sigma^{x,z}` are the pauli matrices. 
The infinite size ground state can be represent by MPS as variational ansatz, where the *virtual bonds* dimension :math:`\chi` effectively controls the number of variational parameters (shown as orange bonds), and the *physical bonds* dimension :math:`d` is the real physical dimension (shown as the blue bonds. Here, for Ising spins :math:`d=2`). 

Because the system has translational invariant, thus it is legit to choose unit-cell consist with two sites, and the infinite system ground state can be represented with only two sites MPS with local tensors :math:`\Gamma_A` and :math:`\Gamma_B` associate with the schmit basis and :math:`\lambda_A`, :math:`\lambda_B` are the diagonal matrices of Schmidt coefficients as shown in the following:

.. image:: image/itebd_MPS.png
    :width: 600
    :align: center

Let's first create this two-site  MPS wave function (here, we set virtual bond dimension :math:`\chi = 10` as example)

* In python

.. code-block:: python
    :linenos:

    import cytnx
    import cytnx.cytnx_extension as cyx

    chi = 10
    A = cyx.CyTensor([cyx.Bond(chi),cyx.Bond(2),cyx.Bond(chi)],rowrank=1,labels=[-1,0,-2])
    B = cyx.CyTensor(A.bonds(),rowrank=1,labels=[-3,1,-4])
    cytnx.random.Make_normal(B.get_block_(),0,0.2)
    cytnx.random.Make_normal(A.get_block_(),0,0.2)
    A.print_diagram()
    B.print_diagram()

    la = cyx.CyTensor([cyx.Bond(chi),cyx.Bond(chi)],rowrank=1,labels=[-2,-3],is_diag=True)
    lb = cyx.CyTensor([cyx.Bond(chi),cyx.Bond(chi)],rowrank=1,labels=[-4,-5],is_diag=True)
    la.put_block(cytnx.ones(chi))
    lb.put_block(cytnx.ones(chi))

    la.print_diagram()
    lb.print_diagram()


* In c++

.. code-block:: c++
    :linenos:

    namespace cyx = cytnx_extension;
    unsigned int chi = 20;
    auto A = cyx::CyTensor({cyx::Bond(chi),cyx::Bond(2),cyx::Bond(chi)},{-1,0,-2},1);
    auto B = cyx::CyTensor(A.bonds(),{-3,1,-4},1);
    cytnx::random::Make_normal(B.get_block_(),0,0.2);
    cytnx::random::Make_normal(A.get_block_(),0,0.2);
    A.print_diagram();
    B.print_diagram();

    auto la = cyx::CyTensor({cyx::Bond(chi),cyx::Bond(chi)},{-2,-3},1,Type.Double,Device.cpu,true);
    auto lb = cyx::CyTensor({cyx::Bond(chi),cyx::Bond(chi)},{-4,-5},1,Type.Double,Device.cpu,true);
    la.put_block(cytnx::ones(chi));
    lb.put_block(cytnx::ones(chi));

    la.print_diagram();
    lb.print_diagram();


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
        -1 ____| 10        2 |____ 0  
               |             |     
               |          10 |____ -2 
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
        -3 ____| 10        2 |____ 1  
               |             |     
               |          10 |____ -4 
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
        -2 ____| 10       10 |____ -3 
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
        -4 ____| 10       10 |____ -5 
               \             /     
                -------------      



Here, we use **random::Make_normal** to initialize the elements of CyTensor *A* and *B* with normal distribution as initial MPS wavefuncion. 
The *la*, *lb* are the weight matrix (schmit coefficients), hence only diagonal elements contains non-zero values. Thus, we set **is_diag=True** to only store diagonal entries. 
We then initialize the elements to be all one for this weight matrices. 

.. Note::
    
    In general, there are other ways you can set-up a trial initial MPS wavefunction, as long as not all the elements are zero. 


Update procedure
******************
To optimize the MPS for the ground state wave function, in TEBD, we perform imaginary time evolution with Hamiltonian :math:`H` with evolution operator :math:`e^{\tau H}`. 
The manybody Hamiltonian is then decomposed into local two-sites evolution operator (or sometimes also called gate in quantum computation language) via 
Trotter-Suzuki decomposition, where :math:`U = e^{\tau H} \approx e^{\delta \tau H_{a}}e^{\delta \tau H_{b}} \cdots = U_a U_b`, :math:`U_{a,b} = e^{\delta \tau H_{a,b}}` are the local evolution operators with :math:`H_a` and :math:`H_b` are the local two sites operator:

.. math::

    H_{a,b} = J\sigma^{z}_{A,B}\sigma^{z}_{B,A} - \frac{H_x}{2}(\sigma^{x}_A + \sigma^{x}_B) 

This is equivalent as acting theses two-site gates consecutively on the MPS, which in terms of tensor notation looks like following Figure(a):

.. image:: image/itebd_upd.png
    :width: 500
    :align: center

Since we represent this infinite system MPS using the translational invariant, the Figure(a) can be further simplified into two step. 
First, acting :math:`U_a` as shown in Figure(1), then acting :math:`U_b` as shown in Figure(2). 

    




.. Note::

    In general, the accurate ground state can be acquired with a higher order Trotter-Suzuki expansion, and with decreasing :math:`\delta \tau` along the iteraction. (See :cite:`itebd-vidal` for further details), Here, for demonstration, we use fixed value of :math:`\delta \tau`. 



.. Hint::
    
    The complete example code can be found in Github repo under example/iTEBD folder.


.. bibliography:: ref.itebd.bib
    :cited:
