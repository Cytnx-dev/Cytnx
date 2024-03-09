import cytnx
import numpy as np

##
# Author: Kai-Hsin Wu
##

#Example of 1D Ising model
## iTEBD
##-------------------------------------

def itebd_tfim(chi = 20, J = 1.0, Hx = 1.0, dt = 0.1, CvgCrit = 1.0e-10):

    ## Create onsite-Op
    Sz = cytnx.physics.pauli("z").real()
    Sx = cytnx.physics.pauli("x").real()
    I = cytnx.eye(2)

    ## Build Evolution Operator
    TFterm = cytnx.linalg.Kron(Sx,I) + cytnx.linalg.Kron(I,Sx)
    ZZterm = cytnx.linalg.Kron(Sz,Sz)

    H = Hx*TFterm + J*ZZterm
    del TFterm, ZZterm
    print(H)

    eH = cytnx.linalg.ExpH(H,-dt) ## or equivantly ExpH(-dt*H)
    eH.reshape_(2,2,2,2)
    print(eH)
    H.reshape_(2,2,2,2)

    eH = cytnx.UniTensor(eH)
    eH.print_diagram()
    print(eH)

    H = cytnx.UniTensor(H)
    H.print_diagram()


    ## Create MPS:
    #
    #     |    |
    #   --A-la-B-lb--
    #
    A = cytnx.UniTensor([cytnx.Bond(chi),cytnx.Bond(2),cytnx.Bond(chi)],labels=['a','0','b']);
    B = cytnx.UniTensor(A.bonds(),rowrank=1,labels=['c','1','d']);
    cytnx.random.normal_(B.get_block_(),0,0.2);
    cytnx.random.normal_(A.get_block_(),0,0.2);
    A.print_diagram()
    B.print_diagram()
    #print(A)
    #print(B)

    la = cytnx.UniTensor([cytnx.Bond(chi),cytnx.Bond(chi)],labels=['b','c'],is_diag=True)
    lb = cytnx.UniTensor([cytnx.Bond(chi),cytnx.Bond(chi)],labels=['d','e'],is_diag=True)
    la.put_block(cytnx.ones(chi));
    lb.put_block(cytnx.ones(chi));
    la.print_diagram()
    lb.print_diagram()
    #print(la)
    #print(lb)

    ## Evov:
    Elast = 0
    for i in range(10000):

        A.set_labels(['a','0','b'])
        B.set_labels(['c','1','d'])
        la.set_labels(['b','c'])
        lb.set_labels(['d','e'])


        ## contract all
        X = cytnx.Contract(cytnx.Contract(A,la),cytnx.Contract(B,lb))
        lb_l = lb.relabel("e","a")
        X = cytnx.Contract(lb_l,X)


        ## X =
        #           (0)  (1)
        #            |    |
        #  (d) --lb-A-la-B-lb-- (e)
        #
        # X.print_diagram()
        Xt = X.clone()


        ## calculate norm and energy for this step
        # Note that X,Xt contract will result a rank-0 tensor, which can use item() toget element
        XNorm = cytnx.Contract(X,Xt).item()
        XH = cytnx.Contract(X,H)
        XH.set_labels(['d','e','0','1'])


        XHX = cytnx.Contract(Xt,XH).item() ## rank-0
        E = XHX/XNorm

        # print(E)
        ## check if converged.
        if(np.abs(E-Elast) < CvgCrit):
            print("[Converged!]")
            break
        print("Step: %d Enr: %5.8f"%(i,Elast))
        Elast = E


        ## Time evolution the MPS
        XeH = cytnx.Contract(X,eH)
        XeH.permute_(['d','2','3','e'])
        # XeH.print_diagram()

        ## Do Svd + truncate
        ##
        #        (2)   (3)                   (2)                                               (3)
        #         |     |          =>         |            + (_aux_L)--s--(_aux_R)  +           |
        #  (d) --= XeH =-- (e)           (d)--U--(_aux_L)                             (_aux_R)--Vt--(e)
        #

        XeH.set_rowrank_(2)
        la,A,B = cytnx.linalg.Svd_truncate(XeH,chi)
        la.normalize_()

        #A.print_diagram()
        #la.print_diagram()
        #B.print_diagram()


        # de-contract the lb tensor , so it returns to
        #             2     3
        #             |     |
        #       d--lb-A'-la-B'-lb--e
        #
        # again, but A' and B' are updated
        lb_inv = 1./lb
        # lb_inv.print_diagram();
        lb_inv.set_labels(['e','d'])

        A = cytnx.Contract(lb_inv,A)
        B = cytnx.Contract(B,lb_inv)

        # translation symmetry, exchange A and B site
        A,B = B,A
        la,lb = lb,la
    return Elast

if __name__ == '__main__':
    itebd_tfim()
