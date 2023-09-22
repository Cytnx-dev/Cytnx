import sys,os
import numpy as np
import cytnx

"""
Reference: https://www.tensors.net
Author: Yu-Hsueh Chen, Kai-Hsin Wu, Hsu Ke j9263178
"""

class Hxx(cytnx.LinOp):

    def __init__(self, anet):
        cytnx.LinOp.__init__(self,"mv", 0, cytnx.Type.Double, cytnx.Device.cpu)
        self.anet = anet
        self.counter = 0

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
    #pshape = [L.shape()[1],M1.shape()[2],M2.shape()[2],R.shape()[1]]

    anet = cytnx.Network()
    anet.FromString(["psi: -1,-2;-3,-4",\
                     "L: -5;-1,0",\
                     "R: -7;-4,3",\
                     "M1: -5,-6;-2,1",\
                     "M2: -6,-7;-3,2",\
                     "TOUT: 0,1;2,3"])
    #anet = cytnx.Network("projectorv2.net")
    anet.PutUniTensor("M2",M2)
    anet.PutUniTensors(["L","M1","R"],[L,M1,R],False)


    H = Hxx(anet)

    energy, psivec = cytnx.linalg.Lanczos_Gnd_Ut(H, Tin=psivec, maxiter = 400, CvgCrit = 1.0e-12)
    #energy ,psivec = cytnx.linalg.Lanczos_Gnd(H,CvgCrit=1.0e-12,Tin=psivec,maxiter=4000)

    return psivec, energy.item()

##### Set bond dimensions and simulation options
chi = 48;
Nsites = 16;
numsweeps = 4 # number of DMRG sweeps
maxit = 2 # iterations of Lanczos method
krydim = 4 # dimension of Krylov subspace

## Initialiaze MPO
##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
d = 2
s = 0.5
#sx = cytnx.physics.spin(0.5,'x')
#sy = cytnx.physics.spin(0.5,'y')
#sp = sx+1j*sy
#sm = sx-1j*sy

#eye = cytnx.eye(d)
#M = cytnx.zeros([4, 4, d, d])
#M[0,0] = M[3,3] = eye
#M[0,1] = M[2,3] = 2**0.5*sp.real()
#M[0,2] = M[1,3] = 2**0.5*sm.real()
#M = cytnx.UniTensor(M,0)

bd_inner = cytnx.Bond(4,cytnx.BD_KET,[[0],[-2],[2],[0]]);
bd_phys = cytnx.Bond(2,cytnx.BD_KET,[[1],[-1]]);

M = cytnx.UniTensor([bd_inner,bd_inner.redirect(),bd_phys, bd_phys.redirect()],rowrank=2)

# I
M.set_elem([0,0,0,0],1);
M.set_elem([0,0,1,1],1);
M.set_elem([3,3,0,0],1);
M.set_elem([3,3,1,1],1);

# S-
M.set_elem([0,1,1,0],2**0.5);

# S+
M.set_elem([0,2,0,1],2**0.5);

# S+
M.set_elem([1,3,0,1],2**0.5);

# S-
M.set_elem([2,3,1,0],2**0.5);


q = 0 # conserving glb Qn
VbdL = cytnx.Bond(1,cytnx.BD_KET,[[0]]);
VbdR = cytnx.Bond(1,cytnx.BD_KET,[[q]]);
L0 = cytnx.UniTensor([bd_inner.redirect(),VbdL.redirect(),VbdL],rowrank=1) #Left boundary
R0 = cytnx.UniTensor([bd_inner,VbdR,VbdR.redirect()],rowrank=1) #Right boundary
L0.set_elem([0,0,0],1); R0.set_elem([3,0,0],1);


## Init MPS train
#
#   0->A[0]->2   2->A[1]->4    4->A[2]->6  ...  2k->A[k]->2k+2
#      |            |             |                 |
#      v            v             v                 v
#      1            3             5                2k+1
#
##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
A = [None for i in range(Nsites)]
qcntr = 0;
if qcntr <= q:
    cq = 1;
else:
    cq = -1;
qcntr+=cq

A[0] = cytnx.UniTensor([VbdL,bd_phys.redirect(),cytnx.Bond(1,cytnx.BD_BRA,[[qcntr]])],rowrank=2)
A[0].get_block_()[0] = 1
for k in range(1,Nsites):
    B1 = A[k-1].bonds()[2].redirect(); B2 = A[k-1].bonds()[1];

    if qcntr <= q:
        cq = 1
    else:
        cq = -1

    qcntr+=cq

    B3 = cytnx.Bond(1,cytnx.BD_BRA,[[qcntr]])

    A[k] = cytnx.UniTensor([B1,B2,B3],rowrank=2)
    A[k].set_labels([2*k,2*k+1,2*k+2])

    A[k].get_block_()[0] = 1

    print(A[k])


# Put in the left normalization form and calculate transfer matrices LR
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

anet = cytnx.Network()
anet.FromString(["L: -2;-1,-3",\
                 "A: -1,-4;1",\
                 "M: -2,0;-4,-5",\
                 "A_Conj: 2;-3,-5",\
                 "TOUT: 0;1,2"])

for p in range(Nsites - 1):
    anet.PutUniTensors(["L","A","A_Conj","M"],[LR[p],A[p],A[p].Dagger(),M],is_clone=False);
    LR[p+1] = anet.Launch(optimal=True);



##checking:
def calMPS(As):
    L = None
    for i in range(len(As)):
        if L is None:
            tA = As[i].relabels([0,1,2])
            L = cytnx.Contract(tA,tA.Dagger().relabel(0,-2))
        else:
            L.set_labels([2,-2])
            tA = As[i].relabels([2,3,4])
            L = cytnx.Contract(tA,L)
            L = cytnx.Contract(L,tA.Dagger().relabels([-4,-2,3]))


    return L.Trace().get_block_().item()


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

        #new_dim = min(dim_l*d,dim_r*d,chi)

        s,A[p],A[p+1] = cytnx.linalg.Svd_truncate(psi,chi)

        tmp = cytnx.Contract(s,s.Dagger())
        s /= tmp.get_block_().item()**0.5


        A[p] = cytnx.Contract(A[p],s) ## absorb s into next neighbor


        # update LR from right to left:
        #anet = cytnx.Network("R_AMAH.net")
        anet = cytnx.Network()
        anet.FromString(["R: -2;-1,-3",\
                         "B: 1;-4,-1",\
                         "M: 0,-2;-4,-5",\
                         "B_Conj: -5,-3;2",\
                         "TOUT: 0;1,2"])
        anet.PutUniTensors(["R","B","M","B_Conj"],[LR[p+2],A[p+1],M,A[p+1].Dagger()],is_clone=False)
        LR[p+1] = anet.Launch(optimal=True)

        print('Sweep[r->l]: %d/%d, Loc:%d,Energy: %f'%(k,numsweeps,p,Ekeep[-1]))

    A[0].set_rowrank(1)
    _,A[0] = cytnx.linalg.Svd(A[0],is_U=False, is_vT=True)


    #A:
    #for rr in A:
    #    rr.print_diagram()
    #exit(1)

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

        #new_dim = min(dim_l*d,dim_r*d,chi)

        s,A[p],A[p+1] = cytnx.linalg.Svd_truncate(psi,chi)

        s /= cytnx.Contract(s,s.Dagger()).get_block_().item()**0.5

        A[p+1] = cytnx.Contract(s,A[p+1]) ## absorb s into next neighbor.



        #anet = cytnx.Network("L_AMAH.net")
        anet = cytnx.Network()
        anet.FromString(["L: -2;-1,-3",\
                         "A: -1,-4;1",\
                         "M: -2,0;-4,-5",\
                         "A_Conj: 2;-3,-5",\
                         "TOUT: 0;1,2"])
        anet.PutUniTensors(["L","A","A_Conj","M"],[LR[p],A[p],A[p].Dagger(),M],is_clone=False);
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
