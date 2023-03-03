import sys
from pathlib import Path
home = str(Path.home())
sys.path.append(home + '/Cytnx_lib')
import cytnx
import numpy as np
import math

##
# Author: Kai-Hsin Wu
##


#Example of 1D Heisenberg model 
## iTEBD
##-------------------------------------

chi = 40
J  = 1.0
CvgCrit = 1.0e-12
dt = 0.1


## Create Si Sj local H with symmetry:
## SzSz + S+S- + h.c. 
bdi = cytnx.Bond(cytnx.BD_KET,[[1],[-1]], [1,1]);
bdo = bdi.clone().set_type(cytnx.BD_BRA);
H = cytnx.UniTensor([bdi.combineBond(bdi,True),bdo.combineBond(bdo,True)],labels=[1,0]);
# H = cytnx.UniTensor([bdi,bdi,bdo,bdo],labels=[2,3,0,1]);

# H.print_diagram()
# H.print_blocks()
# print(bdi.combineBond(bdi,True))
# print(bdi.combineBond(bdi,0))

## assign:
# Q = 2  # Q = 0:    # Q = -2:
# [1]    [[ -1, 1]     [1]
#         [  1,-1]]

# H.get_block_([2])[0] = 1;
# T0 = H.get_block_([0])
# T0[0,0] = T0[1,1] = -1;
# T0[0,1] = T0[1,0] = 1;
# H.get_block_([-2])[0] = 1;
H.get_block_([0,0])[0,0] = 1;
T0 = H.get_block_([1,1])
T0[0,0] = T0[1,1] = -1;
T0[0,1] = T0[1,0] = 1;
H.get_block_([2,2])[0,0] = 1;

## create gate:
eH = cytnx.linalg.ExpH(H,-dt)


## Create MPS:
#
#     |    |     
#   --A-la-B-lb-- 
#
bd_mid = bdi.combineBond(bdi, True);
A = cytnx.UniTensor([bdi,bdi,bd_mid.redirect()],labels=[-1,0,-2]);
B = cytnx.UniTensor([bd_mid,bdi,bdo],labels=[-3,1,-4]);

for b in range(len(B.get_blocks_())):
    cytnx.random.Make_normal(B.get_block_(b),0,0.2);
for a in range(len(A.get_blocks_())):
    cytnx.random.Make_normal(A.get_block_(a),0,0.2);

A.print_diagram()
B.print_diagram()


la = cytnx.UniTensor([bd_mid,bd_mid.redirect()],labels=[-2,-3],is_diag=True)
lb = cytnx.UniTensor([bdi,bdo],labels=[-4,-5],is_diag=True)

for b in range(len(lb.get_blocks_())):
    lb.get_block_(b).fill(1)

for a in range(len(la.get_blocks_())):
    la.get_block_(a).fill(1)

la.print_diagram()
lb.print_diagram()



## Evov:
Elast = 0
for i in range(10000):

    A.set_labels(["-1","0","-2"])
    B.set_labels(["-3","1","-4"])
    la.set_labels(["-2","-3"])
    lb.set_labels(["-4","-5"])

    ## contract all
    X = cytnx.Contract(cytnx.Contract(A,la),cytnx.Contract(B,lb))
    lb.set_label(1,new_label='-1')
    X = cytnx.Contract(lb,X)

    ## X =
    #           (0)  (1)
    #            |    |     
    #  (-4) --lb-A-la-B-lb-- (-5) 
    #
    X.combineBonds(["0","1"],False)
    X.print_diagram()
    X.print_blocks()
    H.print_diagram()
    H.print_blocks()
    # X.set_labels(["-4","0","-5"])
    
    ## calculate local energy:
    ## <psi|psi>
    Xt = X.Dagger()
    XNorm = cytnx.Contract(X,Xt).item()

    ## <psi|H|psi>
    XH = cytnx.Contract(X,H)
    XH.set_labels([-4,-5,0,1])
    XHX = cytnx.Contract(Xt,XH).item()
    
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
    
    ## Do Svd + truncate
    ## 
    #        (2)   (3)                   (2)                                    (3)
    #         |     |          =>         |         +   (-6)--s--(-7)  +         |
    #  (-4) --= XeH =-- (-5)        (-4)--U--(-6)                          (-7)--Vt--(-5)
    #

    XeH.set_rowrank(2)
    if(XeH.shape()[0]*XeH.shape()[1] > chi):
        la,A,B = cytnx.linalg.Svd_truncate(XeH,chi)
    else:
        la,A,B = cytnx.linalg.Svd(XeH)


    Norm = 0;
    for a in range(len(la.get_blocks_())):
        Norm += cytnx.linalg.Norm(la.get_block_(a)).item()**2

    Norm = math.sqrt(Norm)
    for a in range(len(la.get_blocks_())):
        T = la.get_block_(a) 
        T /= Norm
         
    # de-contract the lb tensor , so it returns to 
    #             
    #            |     |     
    #       --lb-A'-la-B'-lb-- 
    #
    # again, but A' and B' are updated 
    A.set_labels([-1,0,-2]); A.set_rowrank(1);
    B.set_labels([-3,1,-4]); B.set_rowrank(1);

    lb_inv = lb.clone()
    for b in range(len(lb_inv.get_blocks_())):
        T = lb_inv.get_block_(b);
        lb_inv.put_block_(1./T,b);

    
 
    A = cytnx.Contract(lb_inv,A)
    B = cytnx.Contract(B,lb_inv)


    # translation symmetry, exchange A and B site
    A,B = B,A
    la,lb = lb,la

