import cytnx
from cytnx import cytnx_extension as cyx
import numpy as np
import scipy as sp
from scipy import linalg

##
# Author: Kai-Hsin Wu
##


#Example of 1D Ising model
## iTEBD
##-------------------------------------

chi = 20
J  = 1.0
Hx = 1.0
CvgCrit = 1.0e-10
dt = 0.1

## Create onsite-Op
Sz = cytnx.zeros([2,2])
Sz[0,0] = 1
Sz[1,1] = -1

Sx = cytnx.zeros([2,2])
Sx[0,1] = Sx[1,0] = Hx

I = Sz.clone()
I[1,1] = 1

#print(Sz,Sx)

## Build Evolution Operator
TFterm = cytnx.linalg.Kron(Sx,I) + cytnx.linalg.Kron(I,Sx)
ZZterm = cytnx.linalg.Kron(Sz,Sz)

H = Hx*TFterm + J*ZZterm
del TFterm, ZZterm

eH = cytnx.linalg.ExpH(H,-dt) ## or equivantly ExpH(-dt*H)
eH.reshape_(2,2,2,2)
print(eH)
H.reshape_(2,2,2,2)

eH = cyx.CyTensor(eH,2)
eH.print_diagram()
print(eH)


H = cyx.CyTensor(H,2)
H.print_diagram()


## Create MPS:
#
#     |    |
#   --A-la-B-lb--
#
A = cyx.CyTensor([cyx.Bond(chi),cyx.Bond(2),cyx.Bond(chi)],rowrank=1,labels=[-1,0,-2]);
B = cyx.CyTensor(A.bonds(),rowrank=1,labels=[-3,1,-4]);
cytnx.random.Make_normal(B.get_block_(),0,0.2);
cytnx.random.Make_normal(A.get_block_(),0,0.2);
A.print_diagram()
B.print_diagram()
#print(A)
#print(B)
la = cyx.CyTensor([cyx.Bond(chi),cyx.Bond(chi)],rowrank=1,labels=[-2,-3],is_diag=True)
lb = cyx.CyTensor([cyx.Bond(chi),cyx.Bond(chi)],rowrank=1,labels=[-4,-5],is_diag=True)
la.put_block(cytnx.ones(chi));
lb.put_block(cytnx.ones(chi));
la.print_diagram()
lb.print_diagram()
#print(la)
#print(lb)

## Evov:
Elast = 0
for i in range(10000):

    A.set_labels([-1,0,-2])
    B.set_labels([-3,1,-4])
    la.set_labels([-2,-3])
    lb.set_labels([-4,-5])

    ## contract all
    X = cyx.Contract(cyx.Contract(A,la),cyx.Contract(B,lb))
    #X.print_diagram()
    lb.set_label(idx=1,new_label=-1)
    X = cyx.Contract(lb,X)

    ## X =
    #           (0)  (1)
    #            |    |
    #  (-4) --lb-A-la-B-lb-- (-5)
    #
    #X.print_diagram()

    Xt = X.clone()

    ## calculate norm and energy for this step
    # Note that X,Xt contract will result a rank-0 tensor, which can use item() toget element
    XNorm = cyx.Contract(X,Xt).item()
    XH = cyx.Contract(X,H)
    XH.set_labels([-4,-5,0,1])
    XHX = cyx.Contract(Xt,XH).item() ## rank-0
    E = XHX/XNorm

    ## check if converged.
    if(np.abs(E-Elast) < CvgCrit):
        print("[Converged!]")
        break
    print("Step: %d Enr: %5.8f"%(i,Elast))
    Elast = E

    ## Time evolution the MPS
    XeH = cyx.Contract(X,eH)
    XeH.permute_([-4,2,3,-5],by_label=True)
    #XeH.print_diagram()

    ## Do Svd + truncate
    ##
    #        (2)   (3)                   (2)                                    (3)
    #         |     |          =>         |         +   (-6)--s--(-7)  +         |
    #  (-4) --= XeH =-- (-5)        (-4)--U--(-6)                          (-7)--Vt--(-5)
    #

    XeH.set_rowrank(2)
    la,A,B = cyx.xlinalg.Svd_truncate(XeH,chi)
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
    A = cyx.Contract(lb_inv,A)
    B = cyx.Contract(B,lb_inv)

    #A.print_diagram()
    #B.print_diagram()

    # translation symmetry, exchange A and B site
    A,B = B,A
    la,lb = lb,la
