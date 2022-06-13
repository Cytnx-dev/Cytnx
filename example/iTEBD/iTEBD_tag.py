import numpy as np
import scipy as sp
from scipy import linalg
import cytnx

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
#
#      ^   ^
#     -2---3-      
#     |  H  |
#     -------
#      ^   ^
#      0   1
#
TFterm = cytnx.linalg.Kron(Sx,I) + cytnx.linalg.Kron(I,Sx)
ZZterm = cytnx.linalg.Kron(Sz,Sz)

H = Hx*TFterm + J*ZZterm 
del TFterm, ZZterm

eH = cytnx.linalg.ExpH(H,-dt) ## or equivantly ExpH(-dt*H)
eH.reshape_(2,2,2,2)
print(eH)
H.reshape_(2,2,2,2)

eH = cytnx.UniTensor(eH,rowrank=2)
eH.tag() # this will tag with in/out(ket/bra) on each bond.
eH.print_diagram()

H = cytnx.UniTensor(H,rowrank=2)
H.tag()
H.print_diagram()


## Create MPS, with bond tagged with direction in/out(ket/bra):
#     ^             ^
#     |             |       
#  ->-A-> ->la->  ->B-> ->lb->
#
A = cytnx.UniTensor([cytnx.Bond(chi,cytnx.BD_KET),cytnx.Bond(2,cytnx.BD_BRA),cytnx.Bond(chi,cytnx.BD_BRA)],rowrank=1,labels=[-1,0,-2]); 
B = cytnx.UniTensor(A.bonds(),rowrank=1,labels=[-3,1,-4]);                                
cytnx.random.Make_normal(B.get_block_(),0,0.2); 
cytnx.random.Make_normal(A.get_block_(),0,0.2); 
A.print_diagram()
B.print_diagram()
#print(A)
#print(B)
la = cytnx.UniTensor([cytnx.Bond(chi,cytnx.BD_KET),cytnx.Bond(chi,cytnx.BD_BRA)],rowrank=1,labels=[-2,-3],is_diag=True)
lb = cytnx.UniTensor(la.bonds(),rowrank=1,labels=[-4,-5],is_diag=True)
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
    X = cytnx.Contract(cytnx.Contract(A,la),cytnx.Contract(B,lb))
    #X.print_diagram()
    lb.set_label(1,new_label=-1)
    X = cytnx.Contract(lb,X)

    ## X =
    #           (0)  (1)
    #            ^    ^     
    #  (-4) ->lb-A-la-B-lb-> (-5) 
    #
    #X.print_diagram()

    Xt = X.Transpose() # it's real type, so we use transpose
    #Xt.print_diagram()
    #exit(1)

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
    #         ^     ^          =>         ^         +   (-6)->s->(-7)  +         ^
    #  (-4) ->= XeH =-> (-5)        (-4)->U->(-6)                          (-7)->Vt->(-5)
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
    #            ^     ^     
    #       ->lb-A'-la-B'-lb-> 
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


        

