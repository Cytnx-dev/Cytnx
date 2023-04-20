iTEBD
------------
**By : Hsu Ke, Kai-Hsin Wu**

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
    A = UniTensor([Bond(chi),Bond(2),Bond(chi)],rowrank=1,labels=[-1,0,-2])
    B = UniTensor(A.bonds(),rowrank=1,labels=[-3,1,-4])
    random.Make_normal(B.get_block_(),0,0.2)
    random.Make_normal(A.get_block_(),0,0.2)
    A.print_diagram()
    B.print_diagram()

    la = UniTensor([Bond(chi),Bond(chi)],rowrank=1,labels=[-2,-3],is_diag=True)
    lb = UniTensor([Bond(chi),Bond(chi)],rowrank=1,labels=[-4,-5],is_diag=True)
    la.put_block(ones(chi))
    lb.put_block(ones(chi))

    la.print_diagram()
    lb.print_diagram()


* In c++

.. code-block:: c++
    :linenos:

    #include "cytnx.hpp"
    using namespace cytnx;

    unsigned int chi = 10;
    auto A = UniTensor({Bond(chi),Bond(2),Bond(chi)},{-1,0,-2},1);
    auto B = UniTensor(A.bonds(),{-3,1,-4},1);
    random::Make_normal(B.get_block_(),0,0.2);
    random::Make_normal(A.get_block_(),0,0.2);
    A.print_diagram();
    B.print_diagram();

    auto la = UniTensor({Bond(chi),Bond(chi)},{-2,-3},1,Type.Double,Device.cpu,true);
    auto lb = UniTensor({Bond(chi),Bond(chi)},{-4,-5},1,Type.Double,Device.cpu,true);
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



Here, we use **random::Make_normal** to initialize the elements of UniTensor *A* and *B* with normal distribution as initial MPS wavefuncion. 
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
    U = UniTensor(eH,2)
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
    // [Note] eH is cytnx.Tensor and U is UniTensor.
    auto eH = linalg::ExpH(H,-dt); //or equivantly ExpH(-dt*H)
    eH.reshape_(2,2,2,2);
    auto U = UniTensor(eH,2);
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

    1. Since :math:`U_a` and :math:`U_b` have the same content(matrix elements) but acting on different sites, we only need to define a single UniTensor. 
    2. Here as a simple example, we directly convert a **cytnx.Tensor** to **cytnx.UniTensor**, which we don't impose any bra-ket constrain (direction of bonds). In general, it is also possible to give bond direction (which we refering to *tagged*) that constrain the bonds to be more physical. See Github example/iTEBD/iTEBD_tag.py for demonstration. 
    3. In general, the accurate ground state can be acquired with a higher order Trotter-Suzuki expansion, and with decreasing :math:`\delta \tau` along the iteraction. (See :cite:`itebd-vidal` for further details), Here, for demonstration, we use fixed value of :math:`\delta \tau`. 
    
.. Tip::

    Here, **physics.pauli** returns complex type **cytnx.Tensor**. Since we know pauli-z and pauli-x should be real, we use *.real()* to get the real part. 


Update procedure
******************
Now we have prepared the initial trial wavefunction in terms of MPS with two sites unit cell and the time evolution operator, we are ready to use the aformentioned scheme to find the (variational) ground state MPS. 
At the beginning of each iteration, we evaluate the energy expectation value :math:`\langle \psi | H | \psi  \rangle / \langle \psi | \psi  \rangle`, and check the convergence, the network is straightforward:


.. image:: image/itebd_contract.png
    :width: 300
    :align: center


.. image:: image/itebd_energy.png
    :width: 450
    :align: center

* In python 

.. code-block:: python 
    :linenos:

    A.set_labels([-1,0,-2])
    B.set_labels([-3,1,-4])
    la.set_labels([-2,-3])
    lb.set_labels([-4,-5])

    ## contract all
    X = cytnx.Contract(cytnx.Contract(A,la),cytnx.Contract(B,lb))
    #X.print_diagram()
    lb.set_label(1,new_label=-1)
    X = cytnx.Contract(lb,X)

    Xt = X.clone()

    ## calculate norm and energy for this step
    # Note that X,Xt contract will result a rank-0 tensor, which can use item() toget element
    XNorm = cytnx.Contract(X,Xt).item()
    XH = cytnx.Contract(X,H)
    XH.set_labels([-4,-5,0,1])
    XHX = cytnx.Contract(Xt,XH).item() ## rank-0
    E = XHX/XNorm

    ## check if converged.
    if(np.abs(E-Elast) < CvgCrit):
        print("[Converged!]")
        break
    print("Step: %d Enr: %5.8f"%(i,Elast))
    Elast = E

* In c++

.. code-block:: c++ 
    :linenos:

    A.set_labels({-1,0,-2}); 
    B.set_labels({-3,1,-4}); 
    la.set_labels({-2,-3}); 
    lb.set_labels({-4,-5}); 


    // contract all
    UniTensor X = cyx::Contract(cyx::Contract(A,la),cyx::Contract(B,lb));
    lb.set_label(1,-1); 
    X = cyx::Contract(lb,X);

    UniTensor Xt = X.clone();
    
    //> calculate norm and energy for this step
    // Note that X,Xt contract will result a rank-0 tensor, which can use item() toget element
    Scalar XNorm = cyx::Contract(X,Xt).item();
    UniTensor XH = cyx::Contract(X,H);

    XH.set_labels({-4,-5,0,1});
    Scalar XHX = cyx::Contract(Xt,XH).item(); 
    double E = double(XHX/XNorm);

    //> check if converged.
    if(abs(E-Elast) < CvgCrit){
        cout << "[Converged!]" << endl;
        break;
    }
    cout << "Step: " << i << "Enr: " << Elast << endl;
    Elast = E;

in the next step we perform the two-sites imaginary time evolution, using the operator (or "gate") eH we defined above:

.. image:: image/itebd_envolve.png
    :width: 700
    :align: center

we also performed SVD for the XeH here, this put the MPS into mixed canonical form and have a Schimit decomposition of the whole state where the singular values are simply the Schimit coefficients. The **Svd_truncate** is called such that the intermediate bonds with label (-6) and (-7) are properly truncate to the maximum virtual bond dimension **chi**. 

* In python 

.. code-block:: python 
    :linenos:

    XeH = cytnx.Contract(X,eH)
    XeH.permute_([-4,2,3,-5],by_label=True)

    XeH.set_rowrank(2)
    la,A,B = cytnx.linalg.Svd_truncate(XeH,chi)
    Norm = cytnx.linalg.Norm(la.get_block_()).item()
    la *= 1./Norm

* In c++

.. code-block:: c++ 
    :linenos:

    //> Time evolution the MPS
    UniTensor XeH = cyx::Contract(X,eH);
    XeH.permute_({-4,2,3,-5},-1,true);

    XeH.set_Rowrank(2);
    vector<UniTensor> out = cyx::xlinalg::Svd_truncate(XeH,chi);
    la = out[0]; A = out[1]; B = out[2];
    Scalar Norm = cytnx::linalg::Norm(la.get_block_()).item();
    la *= 1./Norm; //normalize


Note that we directly store the SVD results into A, B and la, this can be seen by comparing to our original MPS configuration:

.. image:: image/itebd_what.png
    :width: 500
    :align: center

to recover to orignial form, we put :math:`\lambda_B^{-1} \lambda_B` on both ends, which abosorb two :math:`\lambda_B^{-1}`:

.. image:: image/itebd_recover.png
    :width: 500
    :align: center

Now we have the envolved :math:`\Gamma_A`, :math:`\Gamma_B` and :math:`\lambda_A`. Using the translation symmetry, we shift the whole chain to left by just exchange the :math:`Gamma` and :math:`\lambda` pair and arrived at the new MPS for next iteration to update B-A sites using :math:`U_b`. 

.. image:: image/itebd_translation.png
    :width: 300
    :align: center



* In python 

.. code-block:: python 
    :linenos:

    # again, but A' and B' are updated 
    A.set_labels([-1,0,-2]); A.set_rowrank(1);
    B.set_labels([-3,1,-4]); B.set_rowrank(1);

    lb_inv = 1./lb

    lb_inv.set_labels([7, -1]) # -1 to contract with A, 7 is arbitary here.
    A = cytnx.Contract(lb_inv,A)

    lb_inv.set_labels([-4, 8]) # -4 to contract with B, 8 is arbitary here.
    B = cytnx.Contract(B,lb_inv)

    # translation symmetry, exchange A and B site
    A,B = B,A
    la,lb = lb,la

* In c++

.. code-block:: c++ 
    :linenos:

    A.set_labels({-1,0,-2}); A.set_Rowrank(1);
    B.set_labels({-3,1,-4}); B.set_Rowrank(1);
    
    UniTensor lb_inv = 1./lb;

    lb_inv.set_labels({7, -1}); // -1 to contract with A, 7 is arbitary here.
    A = cyx.Contract(lb_inv,A);

    lb_inv.set_labels({-4, 8}) // -4 to contract with B, 8 is arbitary here.
    B = cyx.Contract(B,lb_inv);

    A = cyx::Contract(lb_inv,A);
    B = cyx::Contract(B,lb_inv);


    //> translation symm, exchange A and B site
    UniTensor tmp = A;
    A = B; B = tmp;

    tmp = la;
    la = lb; lb = tmp;

Let's put everything together in a loop for iteration:

* In python 

.. code-block:: python 
    :linenos:

    for i in range(10000):

        A.set_labels([-1,0,-2])
        B.set_labels([-3,1,-4])
        la.set_labels([-2,-3])
        lb.set_labels([-4,-5])

        ## contract all
        X = cytnx.Contract(cytnx.Contract(A,la),cytnx.Contract(B,lb))
        #X.print_diagram()
        lb.set_label(idx=1,new_label=-1)
        X = cytnx.Contract(lb,X)

        ## X =
        #           (0)  (1)
        #            |    |     
        #  (-4) --lb-A-la-B-lb-- (-5) 
        #
        #X.print_diagram()

        Xt = X.clone()

        ## calculate norm and energy for this step
        # Note that X,Xt contract will result a rank-0 tensor, which can use item() toget element
        XNorm = cytnx.Contract(X,Xt).item()
        XH = cytnx.Contract(X,H)
        XH.set_labels([-4,-5,0,1])
        XHX = cytnx.Contract(Xt,XH).item() ## rank-0
        E = XHX/XNorm

        ## check if converged.
        if(np.abs(E-Elast) < CvgCrit):
            print("[Converged!]")
            break
        print("Step: %d Enr: %5.8f"%(i,Elast))
        Elast = E

        ## Time evolution the MPS
        XeH = cytnx.Contract(X,eH)
        XeH.permute_([-4,2,3,-5],by_label=True)
        #XeH.print_diagram()
        
        ## Do Svd + truncate
        ## 
        #        (2)   (3)                   (2)                                    (3)
        #         |     |          =>         |         +   (-6)--s--(-7)  +         |
        #  (-4) --= XeH =-- (-5)        (-4)--U--(-6)                          (-7)--Vt--(-5)
        #

        XeH.set_rowrank(2)
        la,A,B = cytnx.linalg.Svd_truncate(XeH,chi)
        Norm = cytnx.linalg.Norm(la.get_block_()).item()
        la *= 1./Norm
        #A.print_diagram()
        #la.print_diagram()
        #B.print_diagram()
            

        # de-contract the lb tensor , so it returns to 
        #             
        #            |     |     
        #       --lb-A'-la-B'-lb-- 
        #
        # again, but A' and B' are updated 
        A.set_labels([-1,0,-2]); A.set_rowrank(1);
        B.set_labels([-3,1,-4]); B.set_rowrank(1);

        #A.print_diagram()
        #B.print_diagram()

        lb_inv = 1./lb
        A = cytnx.Contract(lb_inv,A)
        B = cytnx.Contract(B,lb_inv)

        #A.print_diagram()
        #B.print_diagram()

        # translation symmetry, exchange A and B site
        A,B = B,A
        la,lb = lb,la


* In c++

.. code-block:: c++ 
    :linenos:
    
    //> Evov:
    double Elast = 0;
    
    for(unsigned int i=0;i<10000;i++){
        A.set_labels({-1,0,-2}); 
        B.set_labels({-3,1,-4}); 
        la.set_labels({-2,-3}); 
        lb.set_labels({-4,-5}); 


        // contract all
        UniTensor X = cyx::Contract(cyx::Contract(A,la),cyx::Contract(B,lb));
        lb.set_label(1,-1); 
        X = cyx::Contract(lb,X);

        UniTensor Xt = X.clone();
        
        //> calculate norm and energy for this step
        // Note that X,Xt contract will result a rank-0 tensor, which can use item() toget element
        double XNorm = cyx::Contract(X,Xt).item<double>();
        UniTensor XH = cyx::Contract(X,H);

        XH.set_labels({-4,-5,0,1});
        double XHX = cyx::Contract(Xt,XH).item<double>(); 
        double E = XHX/XNorm;

        //> check if converged.
        if(abs(E-Elast) < CvgCrit){
            cout << "[Converged!]" << endl;
            break;
        }
        cout << "Step: " << i << "Enr: " << Elast << endl;
        Elast = E;

        //> Time evolution the MPS
        UniTensor XeH = cyx::Contract(X,eH);
        XeH.permute_({-4,2,3,-5},-1,true);

        //> Do Svd + truncate
        XeH.set_Rowrank(2);
        vector<UniTensor> out = cyx::xlinalg::Svd_truncate(XeH,chi);
        la = out[0]; A = out[1]; B = out[2];
        double Norm = cytnx::linalg::Norm(la.get_block_()).item<double>();
        la *= 1./Norm; //normalize
        

        // de-contract the lb tensor , so it returns to 
        //             
        //            |     |     
        //       --lb-A'-la-B'-lb-- 
        //
        // again, but A' and B' are updated 
        A.set_labels({-1,0,-2}); A.set_Rowrank(1);
        B.set_labels({-3,1,-4}); B.set_Rowrank(1);
        
        UniTensor lb_inv = 1./lb;
        A = cyx::Contract(lb_inv,A);
        B = cyx::Contract(B,lb_inv);

    
        //> translation symm, exchange A and B site
        UniTensor tmp = A;
        A = B; B = tmp;

        tmp = la;
        la = lb; lb = tmp;
    }

.. bibliography:: ref.itebd.bib
    :cited:
