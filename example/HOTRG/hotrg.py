import cytnx 
from  cytnx import linalg as cLA
import numpy as np

"""
References: https://arxiv.org/pdf/1201.1144.pdf
            https://journals.aps.org/prb/pdf/10.1103/PhysRevB.81.174411
Author: Kai-Hsin Wu 
"""

## 2D Ising model:
chi = 8
Cvg_crit = 1.0e-12
Maxiter = 100

## init the local tensor:
beta = 5
W = cytnx.zeros([2,2])
W[0,0] = W[1,0] = np.sqrt(np.cosh(beta)) 
W[0,1] = np.sqrt(np.sinh(beta)) 
W[1,1] = -np.sqrt(np.sinh(beta)) 

## Method-1 (faster): 
T = cLA.Kron(cLA.Kron(W[0],W[0]),cLA.Kron(W[0],W[0]))+\
    cLA.Kron(cLA.Kron(W[1],W[1]),cLA.Kron(W[1],W[1])) 
T.reshape_(2,2,2,2)

## Method-2 (slower in python):
#Tchk = cytnx.zeros([2,2,2,2])
#for i in range(2):
#    for j in range(2):
#        for k in range(2):
#            for l in range(2):
#                for a in range(2):
#                    Tchk[i,j,k,l] += W[a,i]*W[a,j]*W[a,k]*W[a,l]
cT = cytnx.UniTensor(T,rowrank=2)


## Let's start by normalize the block with it's local partition function 
## to avoid the blow-up of partition function:
#
#         1
#         |  
#     0--cT--2
#         |
#         3
#
Nrm = cT.get_block().Trace(0,2).Trace(0,1).item()
cT/=Nrm




nrm_factor =[] ## Now, we want to keep track the normalization factor used in each iteration step
prevlnZ = 0;
for i in range(Maxiter):
    print(i)


    ## LR:
    #
    #         1           4
    #         |           |
    #     0--cT--2    2--cT2--5
    #         |           |
    #         3           6
    #
    cT2 = cT.clone()
    cT2.set_labels([2,4,5,6])
    cT = cytnx.Contract(cT,cT2)

    ## Now, let's check the dimension growth onto a point where truncation is needed:
    if(cT.shape()[1]*cT.shape()[3]>chi):
        # * if combined bond dimension > chi then:
        # 1) Do Hosvd get only U and D, with it's Ls matrices. 
        cT.permute_([1,4,3,6,0,5],by_label=True)
        U,D,Lu,Ld=cytnx.linalg.Hosvd(cT,[2,2],is_core=False,is_Ls=True)

        # 2) Using Ls matrix to determine if U is used to truncate or D is used to truncate
        if(Lu.get_block_()[chi:].Norm().item() < Ld.get_block_()[chi:].Norm().item()):
            U, D = D,U

        ## truncate, and permute back to the original form:
        #              chi
        #               |
        #              U/D 
        #             /   \
        #            |     |
        #           (d)   (d)                    (chi)
        #            1     4                       1 
        #            |     |                       | 
        #     (d) 0--[ cT  ]--5 (d)   =>   (d) 0--cT--2 (d)
        #            |     |                       |
        #            3     6                       3
        #           (d)   (d)                    (chi)
        #             \   /
        #              U/D
        #               |  
        #              chi
        # [Note] here "d" is used to indicate the original bond dimension of each rank, 
        #        in general, they could be different for each bond
        U.truncate_(2,chi);

        cT = cytnx.Contract(cT,U)
        U.set_labels(D.labels())
        cT = cytnx.Contract(cT,U)

        ## set back to the original shape:
        cT.set_labels([1,3,0,2])
        cT.permute_([0,1,2,3],by_label=True)
        cT.set_rowrank(2)

    else: 
        # * if combined bond dimension <= chi then we just combined the bond, and return it's original form
        #           (d)  (d)                    (dxd)
        #            1    4                       1 
        #            |    |                       | 
        #    (d) 0--[cT.cT2]--5 (d)   =>   (d)0--cT--2 (d)
        #            |    |                       |
        #            3    6                       3
        #           (d)  (d)                    (dxd)
        #
        # [Note] here "d" is used to indicate the original bond dimension of each rank, 
        #        in general, they could be different for each bond
        cT.permute_([0,1,4,5,3,6],by_label=True)
        cT.contiguous_()
        cT.reshape_(cT.shape()[0],cT.shape()[1]*cT.shape()[2],cT.shape()[3],cT.shape()[4]*cT.shape()[5])
        cT.set_rowrank(2)



    ## UD:
    #
    #         1           
    #         |           
    #     0--cT--2    
    #         |           
    #         3           
    #
    #         3           
    #         |           
    #     6--cT2--4
    #         |           
    #         5           
    #
    #
    cT2 = cT.clone()
    cT2.set_labels([6,3,4,5])
    cT = cytnx.Contract(cT,cT2)
    
    ## check the dimension growth onto a point where truncation is needed:
    if(cT.shape()[2]*cT.shape()[4]>chi):
        # * if combined bond dimension > chi then:
        # 1) Do Hosvd get only L and R, with it's Ls matrices. 
        cT.permute_([2,4,0,6,1,5],by_label=True)
        L,R,Ll,Lr=cytnx.linalg.Hosvd(cT,[2,2],is_core=False,is_Ls=True)

        # 2) Using Ls matrix to determine if L is used to truncate or R is used to truncate
        if(Ll.get_block_()[chi:].Norm().item() < Lr.get_block_()[chi:].Norm().item()):
            L,R = R,L

        ## truncate, and permute back to the original form:
        #               (d)
        #                1           
        #                |                              (d)
        #            0--cT --2                           1  
        #           /    |    \                          |
        # (chi) --L/R    |    L/R-- (chi)  =>  (chi) 0--cT--2 (chi)
        #           \    |    /                          |
        #            6--cT2--4                          (d)
        #                |           
        #                5           
        #               (d)
        #
        # [Note] here "d" is used to indicate the original bond dimension of each rank, 
        #        in general, they could be different for each bond
        L.truncate_(2,chi);

        cT = cytnx.Contract(cT,L)
        L.set_labels(R.labels())
        cT = cytnx.Contract(cT,L)

        ## set back to the original shape:
        cT.set_labels([1,3,2,0])
        cT.permute_([0,1,2,3],by_label=True)
        cT.set_rowrank(2)
    else:
        # * if combined bond dimension <= chi then we just combined the bond, and return it's original form
        #
        #         (d)
        #          1                                (d)
        #          |                                 1
        #  (d) 0--cT --2 (d)                         |
        #          |              =>       (dxd) 0--cT--2 (dxd) 
        #  (d) 6--cT2--4 (d)                         |
        #          |                                 3
        #          5                                (d)
        #         (d)
        #
        # [Note] here "d" is used to indicate the original bond dimension of each rank, 
        #        in general, they could be different for each bond
        cT.permute_([0,6,1,2,4,5],by_label=True)
        cT.contiguous_()
        cT.reshape_(cT.shape()[0]*cT.shape()[1],cT.shape()[2],cT.shape()[3]*cT.shape()[4],cT.shape()[5])
        cT.set_rowrank(2)


    #normalize:
    Z = cT.get_block().Trace(0,2)
    Z = Z.Trace(0,1)
    Nrm = Z.item()
    cT /= Nrm
    nrm_factor.append(np.log(Nrm))


    ## let's calculate the free energy per-site:
    # Using the previous memorized normalization factor on each step.
    lnZ = 0;
    for n in range(len(nrm_factor)):
        lnZ += nrm_factor[-1-n]*4**(n-i-1);
    F = -lnZ/beta
    print("F/N:",F)

    ## if lnZ per-site is converged, then we reach the fix point.
    if(np.abs(lnZ-prevlnZ)<Cvg_crit):
        break;

    prevlnZ = lnZ;

    
print("[Converged!]")


