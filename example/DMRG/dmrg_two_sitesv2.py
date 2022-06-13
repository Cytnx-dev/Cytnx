import sys,os
import numpy as np
import cytnx


"""
Reference: https://www.tensors.net
Author: Yu-Hsueh Chen, Kai-Hsin Wu, Hsu Ke j9263178
"""

class Hxx(cytnx.LinOp):

    def __init__(self, anet, shapes):
        cytnx.LinOp.__init__(self,"mv", 0, cytnx.Type.Double, cytnx.Device.cpu)
        self.anet = anet
        self.shapes = shapes

    def matvec(self, v):
        lbl = v.labels(); 
        self.anet.PutUniTensor("psi",v);
        out = self.anet.Launch(optimal=True)
        out.set_labels(lbl); ##make sure the input label match output label!!
        out.contiguous_();
        return out

def optimize_psi(psivec, functArgs, maxit=2, krydim=4):

    """ Lanczos method for finding smallest algebraic eigenvector of linear \
    operator defined as a function"""
    #print(eig_Lanczos)
    #create network!
    L,M1,M2,R = functArgs
    pshape = [L.shape()[1],M1.shape()[2],M2.shape()[2],R.shape()[1]]

    anet = cytnx.Network("projectorv2.net")
    anet.PutUniTensor("M2",M2)
    anet.PutUniTensors(["L","M1","R"],[L,M1,R],False)


    H = Hxx(anet, pshape)
    energy, psivec = cytnx.linalg.Lanczos_Gnd_Ut(H, Tin=psivec, maxiter = 400, CvgCrit = 1.0e-12)
    #energy ,psivec = cytnx.linalg.Lanczos_Gnd(H,CvgCrit=1.0e-12,Tin=psivec,maxiter=4000)
 
    return psivec, energy.item()

##### Set bond dimensions and simulation options
chi = 32;
Nsites = 4;
numsweeps = 4 # number of DMRG sweeps
maxit = 2 # iterations of Lanczos method
krydim = 4 # dimension of Krylov subspace

## Initialiaze MPO 
##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
d = 2
s = 0.5
sx = cytnx.physics.spin(0.5,'x')
sy = cytnx.physics.spin(0.5,'y')
sp = sx+1j*sy
sm = sx-1j*sy

eye = cytnx.eye(d)
M = cytnx.zeros([4, 4, d, d])
M[0,0] = M[3,3] = eye
M[0,1] = M[2,3] = 2**0.5*sp.real()
M[0,2] = M[1,3] = 2**0.5*sm.real()
M = cytnx.UniTensor(M,rowrank=0)

L0 = cytnx.UniTensor(cytnx.zeros([4,1,1]),rowrank=0) #Left boundary
R0 = cytnx.UniTensor(cytnx.zeros([4,1,1]),rowrank=0) #Right boundary
L0.get_block_()[0,0,0] = 1.; R0.get_block_()[3,0,0] = 1.

## Init MPS train
#   
#   0-A[0]-2    2-A[1]-4    4-A[2]-6  ...  2k-A[k]-2k+2
#      |           |           |               |
#      1           3           5              2k+1
#
##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
A = [None for i in range(Nsites)]
A[0] = cytnx.UniTensor(cytnx.random.normal([1, d, min(chi, d)], 0., 1.),rowrank=2)
for k in range(1,Nsites):
    dim1 = A[k-1].shape()[2]; dim2 = d;
    dim3 = min(min(chi, A[k-1].shape()[2] * d), d ** (Nsites - k - 1));
    A[k] = cytnx.UniTensor(cytnx.random.normal([dim1, dim2, dim3],0.,1.),rowrank=2)
    A[k].set_labels([2*k,2*k+1,2*k+2])
    A[k].print_diagram()

## Put in the left normalization form and calculate transfer matrices LR
#LR[0]:        LR[1]:            LR[2]:
#
#   -----      -----A[0]---     -----A[1]---
#   |          |     |          |     |
#  ML----     LR[0]--M-----    LR[1]--M-----      ......
#   |          |     |          |     |
#   -----      -----A*[0]--     -----A*[1]--
#
#
# L_AMAH.net file is used to contract the LR[i]
##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LR = [None for i in range(Nsites+1)]
LR[0]  = L0
LR[-1] = R0

for p in range(Nsites - 1):
    s, A[p] ,vt = cytnx.linalg.Svd(A[p])
    A[p+1] = cytnx.Contract(cytnx.Contract(s,vt),A[p+1])
    #A[p+1].print_diagram()
    #A[p].print_diagram()
    anet = cytnx.Network("L_AMAH.net")
    anet.PutUniTensors(["L","A","A_Conj","M"],[LR[p],A[p],A[p].Conj(),M],is_clone=False);
    LR[p+1] = anet.Launch(optimal=True);

            
_,A[-1] = cytnx.linalg.Svd(A[-1],is_U=True,is_vT=False) ## last one.


## DMRG sweep
## Now we are ready for sweeping procedure!
##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Ekeep = []

for k in range(1, numsweeps+2):
    
    # a. Optimize from right-to-left:
    # psi:                   Projector:
    # 
    #   --A[p]--A[p+1]--s--              --         --
    #      |       |                     |    | |    |
    #                                   LR[p]-M-M-LR[p+1]
    #                                    |    | |    |
    #                                    --         --
    # b. Transfer matrix from right to left :
    #  LR[-1]:       LR[-2]:            
    #
    #      ---          ---A[-1]---         
    #        |               |    | 
    #      --MR         -----M--LR[-1]   ......
    #        |               |    |
    #      ---          ---A*[-1]--
    #
    # c. For Right to Left, we want A's to be in shape
    #            -------------      
    #           /             \     
    #  virt ____| chi     chi |____ virt
    #           |             |     
    #  phys ____| 2           |        
    #           \             /     
    #            -------------      



    for p in range(Nsites-2,-1,-1): 

        dim_l = A[p].shape()[0];
        dim_r = A[p+1].shape()[2];


        psi = cytnx.Contract(A[p],A[p+1]) ## contract
        psi.set_rowrank(2);
        psi, Entemp = optimize_psi(psi, (LR[p],M,M,LR[p+2]), maxit, krydim)
        Ekeep.append(Entemp);
        
        new_dim = min(dim_l*d,dim_r*d,chi)

        s,A[p],A[p+1] = cytnx.linalg.Svd_truncate(psi,new_dim)

        # s = s.Div(s.get_block_().Norm().item()) 
        # s.Div_(s.get_block_().Norm().item()) // a bug : cannot use
        slabel = s.labels()
        s = s/s.get_block_().Norm().item() 
        s.set_labels(slabel)


        A[p] = cytnx.Contract(A[p],s) ## absorb s into next neighbor

        # A[p].print_diagram()
        # A[p+1].print_diagram()

        # update LR from right to left:
        anet = cytnx.Network("R_AMAH.net")
        anet.PutUniTensors(["R","B","M","B_Conj"],[LR[p+2],A[p+1],M,A[p+1].Conj()],is_clone=False)
        LR[p+1] = anet.Launch(optimal=True)
        
        print('Sweep[r->l]: %d/%d, Loc:%d,Energy: %f'%(k,numsweeps,p,Ekeep[-1]))

    A[0].set_rowrank(1)
    _,A[0] = cytnx.linalg.Svd(A[0],is_U=False, is_vT=True)

    # a.2 Optimize from left-to-right:
    # psi:                   Projector:
    # 
    #   --A[p]--A[p+1]--s--              --         --
    #      |       |                     |    | |    |
    #                                   LR[p]-M-M-LR[p+1]
    #                                    |    | |    |
    #                                    --         --
    # b.2 Transfer matrix from left to right :
    #  LR[0]:       LR[1]:                   
    #
    #      ---          ---A[0]---                 
    #      |            |    |     
    #      L0-         LR[0]-M----    ......
    #      |            |    |
    #      ---          ---A*[0]--
    #
    # c.2 For Left to Right, we want A's to be in shape
    #            -------------      
    #           /             \     
    #  virt ____| chi     2   |____ phys
    #           |             |     
    #           |        chi  |____ virt        
    #           \             /     
    #            -------------      

    for p in range(Nsites-1):
        dim_l = A[p].shape()[0]
        dim_r = A[p+1].shape()[2]

        psi = cytnx.Contract(A[p],A[p+1]) ## contract
        psi.set_rowrank(2);
        psi, Entemp = optimize_psi(psi, (LR[p],M,M,LR[p+2]), maxit, krydim)
        Ekeep.append(Entemp);
        
        new_dim = min(dim_l*d,dim_r*d,chi)

        s,A[p],A[p+1] = cytnx.linalg.Svd_truncate(psi,new_dim)

        # s = s/s.get_block_().Norm().item()
        slabel = s.labels()
        s = s/s.get_block_().Norm().item() 
        s.set_labels(slabel)

        A[p+1] = cytnx.Contract(s,A[p+1]) ## absorb s into next neighbor.

        anet = cytnx.Network("L_AMAH.net")
        anet.PutUniTensors(["L","A","A_Conj","M"],[LR[p],A[p],A[p].Conj(),M],is_clone=False);
        LR[p+1] = anet.Launch(optimal=True);

        print('Sweep[l->r]: %d of %d, Loc: %d,Energy: %f' % (k, numsweeps, p, Ekeep[-1]))

    A[-1].set_rowrank(2)
    _,A[-1] = cytnx.linalg.Svd(A[-1],is_U=True,is_vT=False) ## last one.
    print('done : %d'% k)




#### Compare with exact results (computed from free fermions)
from numpy import linalg as LA
# import matplotlib.pyplot as plt
H = np.diag(np.ones(Nsites-1),k=1) + np.diag(np.ones(Nsites-1),k=-1)
D = LA.eigvalsh(H)
EnExact = 2*sum(D[D < 0])
print("Exact : ")
print(EnExact)

##### Plot results
# plt.figure(1)
# plt.yscale('log')
# plt.plot(range(len(Ekeep)), np.array(Ekeep) - EnExact, 'b', label="chi = %d"%(chi), marker = 'o')
# plt.legend()
# plt.title('DMRG for XX model')
# plt.xlabel('Update Step')
# plt.ylabel('Ground Energy Error')
# plt.show()



