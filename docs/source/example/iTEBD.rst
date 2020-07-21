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

    from cytnx import *

    chi = 10
    A = CyTensor([Bond(chi),Bond(2),Bond(chi)],rowrank=1,labels=[-1,0,-2])
    B = CyTensor(A.bonds(),rowrank=1,labels=[-3,1,-4])
    random.Make_normal(B.get_block_(),0,0.2)
    random.Make_normal(A.get_block_(),0,0.2)
    A.print_diagram()
    B.print_diagram()

    la = CyTensor([Bond(chi),Bond(chi)],rowrank=1,labels=[-2,-3],is_diag=True)
    lb = CyTensor([Bond(chi),Bond(chi)],rowrank=1,labels=[-4,-5],is_diag=True)
    la.put_block(ones(chi))
    lb.put_block(ones(chi))

    la.print_diagram()
    lb.print_diagram()


* In c++

.. code-block:: c++
    :linenos:

    #include "cytnx.hpp"
    using namespace cytnx;

    unsigned int chi = 20;
    auto A = CyTensor({Bond(chi),Bond(2),Bond(chi)},{-1,0,-2},1);
    auto B = CyTensor(A.bonds(),{-3,1,-4},1);
    random::Make_normal(B.get_block_(),0,0.2);
    random::Make_normal(A.get_block_(),0,0.2);
    A.print_diagram();
    B.print_diagram();

    auto la = CyTensor({Bond(chi),Bond(chi)},{-2,-3},1,Type.Double,Device.cpu,true);
    auto lb = CyTensor({Bond(chi),Bond(chi)},{-4,-5},1,Type.Double,Device.cpu,true);
    la.put_block(ones(chi));
    lb.put_block(ones(chi));

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


Imaginary time evolution
*************************
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
First, acting :math:`U_a` as shown in Figure(1) then acting :math:`U_b` as shown in Figure(2). This two procedures then repeat until the energy is converged. 

Here, let's construct this imaginary time evolution operator with parameter :math:`J=-1`, :math:`H_x = -0.3` and (imaginary) time step :math:`\delta \tau = 0.1`

* In python 

.. code-block:: python 
    :linenos:

    J = -1.0
    Hx = -0.3
    dt = 0.1

    ## Create single site operator
    Sz = physics.pauli('z').real()
    Sx = physics.pauli('x').real()
    I  = eye(2)
    print(Sz)
    print(Sx)


    ## Construct the local Hamiltonian
    TFterm = linalg.Kron(Sx,I) + linalg.Kron(I,Sx)
    ZZterm = linalg.Kron(Sz,Sz)
    H = Hx*TFterm + J*ZZterm
    print(H)


    ## Build Evolution Operator
    eH = linalg.ExpH(H,-dt) ## or equivantly ExpH(-dt*H)
    eH.reshape_(2,2,2,2)
    U = CyTensor(eH,2)
    U.print_diagram()


* In c++

.. code-block:: c++
    :linenos:

    double J = -1.0;
    double Hx = -0.3;
    double dt = 0.1;

    // Create single site operator
    auto Sz = physics::pauli('z').real();
    auto Sx = physics::pauli('x').real();
    auto I  = eye(2);
    cout << Sz << endl;
    cout << Sx << endl;


    // Construct the local Hamiltonian
    auto TFterm = linalg::Kron(Sx,I) + linalg::Kron(I,Sx);
    auto ZZterm = linalg::Kron(Sz,Sz);
    auto H = Hx*TFterm + J*ZZterm;
    cout << H << endl;


    // Build Evolution Operator
    // [Note] eH is cytnx.Tensor and U is CyTensor.
    auto eH = linalg::ExpH(H,-dt); //or equivantly ExpH(-dt*H)
    eH.reshape_(2,2,2,2);
    auto U = CyTensor(eH,2);
    U.print_diagram();

Output>>

.. code-block:: text

    Total elem: 4
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (2,2)
    [[1.00000e+00 0.00000e+00 ]
     [0.00000e+00 -1.00000e+00 ]]


    Total elem: 4
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (2,2)
    [[0.00000e+00 1.00000e+00 ]
     [1.00000e+00 0.00000e+00 ]]


    Total elem: 16
    type  : Double (Float64)
    cytnx device: CPU
    Shape : (4,4)
    [[-1.00000e+00 3.00000e-01 3.00000e-01 0.00000e+00 ]
     [3.00000e-01 1.00000e+00 0.00000e+00 3.00000e-01 ]
     [3.00000e-01 0.00000e+00 1.00000e+00 3.00000e-01 ]
     [0.00000e+00 3.00000e-01 3.00000e-01 -1.00000e+00 ]]

    -----------------------
    tensor Name : 
    tensor Rank : 4
    block_form  : false
    is_diag     : False
    on device   : cytnx device: CPU
                -------------      
               /             \     
         0 ____| 2         2 |____ 2  
               |             |     
         1 ____| 2         2 |____ 3  
               \             /     
                -------------      




.. Note::

    1. Since :math:`U_a` and :math:`U_b` have the same content(matrix elements) but acting on different sites, we only need to define a single CyTensor. 
    2. Here as a simple example, we directly convert a **cytnx.Tensor** to **cyx.CyTensor**, which we don't impose any bra-ket constrain (direction of bonds). In general, it is also possible to give bond direction (which we refering to *tagged*) that constrain the bonds to be more physical. See Github example/iTEBD/iTEBD_tag.py for demonstration. 
    3. In general, the accurate ground state can be acquired with a higher order Trotter-Suzuki expansion, and with decreasing :math:`\delta \tau` along the iteraction. (See :cite:`itebd-vidal` for further details), Here, for demonstration, we use fixed value of :math:`\delta \tau`. 
    
.. Tip::

    Here, **physics.pauli** returns complex type **cytnx.Tensor**. Since we know pauli-z and pauli-x should be real, we use *.real()* to get the real part. 


Update procedure
******************
Now we have prepared the initial trial wavefunction in terms of MPS with two sites unit cell and the time evolution operator, we are ready to use the aformentioned scheme to find the (variational) ground state MPS. 



.. Hint::
    
    The complete example code can be found in Github repo under example/iTEBD folder.


.. bibliography:: ref.itebd.bib
    :cited:
