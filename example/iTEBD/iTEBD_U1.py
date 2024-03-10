import os,sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home + '/Cytnx_lib')
import cytnx
import math
from cytnx import Qs,BD_IN, BD_OUT
import numpy as np

##
# Author: Kai-Hsin Wu
##


#Example of 1D Heisenberg model
## iTEBD
##-------------------------------------

def itebd_heisenberg(chi = 32, J  = 1.0, dt = 0.1, CvgCrit = 1.0e-12):


    ## Create Si Sj local H with symmetry:
    ## SzSz + S+S- + h.c.
    bdi = cytnx.Bond(BD_IN,[Qs(1)>>1,Qs(-1)>>1]);
    bdo = bdi.clone().set_type(BD_OUT);
    H = cytnx.UniTensor([bdi,bdi,bdo,bdo],labels=['2','3','1','0']);

    ## assign:
    H.at([0,0,0,0]).value = 1;
    H.at([0,1,0,1]).value = -1;
    H.at([1,0,1,0]).value = -1;
    H.at([0,1,1,0]).value = 1;
    H.at([1,0,0,1]).value = 1;
    H.at([1,1,1,1]).value = 1;

    ## create gate:
    eH = cytnx.linalg.ExpH(H,-dt)


    ## Create MPS:
    #
    #     |    |
    #   --A-la-B-lb--
    #
    bd_mid = bdi.combineBond(bdi, True);
    A = cytnx.UniTensor([bdi,bdi,bd_mid.redirect()],labels=['a','0','b']);
    B = cytnx.UniTensor([bd_mid,bdi,bdo],labels=['c','1','d']);

    for b in range(len(B.get_blocks_())):
        cytnx.random.normal_(B.get_block_(b),0,0.2);
    for a in range(len(A.get_blocks_())):
        cytnx.random.normal_(A.get_block_(a),0,0.2);

    A.print_diagram()
    B.print_diagram()


    la = cytnx.UniTensor([bd_mid,bd_mid.redirect()],labels=['b','c'],is_diag=True)
    lb = cytnx.UniTensor([bdi,bdo],labels=['d','e'],is_diag=True)

    for b in range(len(lb.get_blocks_())):
        lb.get_block_(b).fill(1)

    for a in range(len(la.get_blocks_())):
        la.get_block_(a).fill(1)

    la.print_diagram()
    lb.print_diagram()



    ## Evov:
    Elast = 0
    for i in range(10000):

        A.set_labels(["a","0","b"])
        B.set_labels(["c","1","d"])
        la.set_labels(["b","c"])
        lb.set_labels(["d","e"])

        ## contract all
        tmpA = cytnx.Contract(A,la)
        tmpB = cytnx.Contract(B,lb)
        X = cytnx.Contract(tmpA,tmpB);# << "this line cause problem!\n";
        #X = cytnx.Contract(cytnx.Contract(A,la),cytnx.Contract(B,lb))
        #exit(1)
        lb.set_label("e",new_label='a')
        X = cytnx.Contract(lb,X)


        ## X =
        #           (0)  (1)
        #            |    |
        #  (d) --lb-A-la-B-lb-- (e)
        #
        ## calculate local energy:
        ## <psi|psi>
        Xt = X.Dagger()
        XNorm = cytnx.Contract(X,Xt).item()

        ## <psi|H|psi>
        XH = cytnx.Contract(X,H)
        #XH.print_diagram()
        XH.set_labels(['d','e','0','1'])
        XHX = cytnx.Contract(Xt,XH).item()
        E = XHX/XNorm



        ## check if converged.
        if(abs(E-Elast) < CvgCrit):
            print("[Converged!]")
            break
        print("Step: %d Enr: %5.8f"%(i,Elast))
        Elast = E


        ## Time evolution the MPS
        XeH = cytnx.Contract(X,eH)
        XeH.permute_(['d','2','3','e'])

        ## Do Svd + truncate
        ##
        #       (2)   (3)                 (2)                                                (3)
        #        |     |          =>       |         +   (_aux_L)--s--(_aux_R)  +             |
        #  (d) --= XeH =-- (e)        (d)--U--(_aux_L)                              (_aux_R)--Vt--(e)
        #

        XeH.set_rowrank_(2)
        la,A,B = cytnx.linalg.Svd_truncate(XeH,chi)
        la.normalize_()

        # de-contract the lb tensor , so it returns to
        #
        #            |     |
        #       --lb-A'-la-B'-lb--
        #
        # again, but A' and B' are updated
        lb_inv = lb.clone()
        for b in range(len(lb_inv.get_blocks_())):
            T = lb_inv.get_block_(b);
            lb_inv.put_block_(1./T,b);

        lb_inv.set_labels(['e','d'])

        A = cytnx.Contract(lb_inv,A)
        B = cytnx.Contract(B,lb_inv)


        # translation symmetry, exchange A and B site
        A,B = B,A
        la,lb = lb,la
    return Elast

if __name__ == '__main__':
    itebd_heisenberg()
