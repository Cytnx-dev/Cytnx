import sys,os
import numpy as np
import cytnx

import time

"""
Reference: https://www.tensors.net
Author: Yu-Hsueh Chen, Kai-Hsin Wu, Hsu Ke j9263178
"""

Lanc_time = []
HAA_time = []
AA_time = []
sweep_time = []

XXExact = -40.03277580097028

def print_bond_qn(qns):
    import numpy as np
    tamed = np.asarray(qns).transpose()[0]
    uniqs = set(tamed)
    for ele in uniqs:
        count = 0
        for qn in tamed:
            if qn == ele:
                count += 1
        print("(",ele,",",count,end=') ')

def print_bond_dir(ut):
    bds = ut.bonds()
    for bd in bds:
        print(bd.type(), end = " ")
    print("")


class Hxx(cytnx.LinOp):

    def __init__(self, anet,L,M1,M2,R):
        cytnx.LinOp.__init__(self,"mv", 0, cytnx.Type.Double, cytnx.Device.cpu)
        self.anet = anet
        self.counter = 0
        self.L = L
        self.R = R
        self.M1 = M1
        self.M2 = M2
    def matvec(self, v):
        start = time.time()

        lbl = v.labels();
        # print(lbl)
        # self.anet.PutUniTensor("psi",v);
        L_ = self.L.relabels([-5,-1,0]);
        R_ = self.R.relabels([-7,-4,3]);
        M1_ = self.M1.relabels([-5,-6,-2,1]);
        M2_ = self.M2.relabels([-6,-7,-3,2]);
        psi_ = v.relabels([-1,-2,-3,-4]);
        out = L_.contract(M1_.contract(M2_.contract(psi_.contract(R_, False, False), False, False), False, False), False, False)
        # out = self.anet.Launch(True);

        # out.set_rowrank(2)
        out.contiguous_()

        out.set_labels(lbl) ##make sure the input label match output label!!

        end = time.time()

        if(v.shape()==[chi, 2,2, chi]):
            HAA_time.append(end-start)
        return out

def optimize_psi(psivec, functArgs, maxit, krydim=4):

    """ Lanczos method for finding smallest algebraic eigenvector of linear \
    operator defined as a function"""
    #print(eig_Lanczos)
    #create network!
    L,M1,M2,R = functArgs
    #pshape = [L.shape()[1],M1.shape()[2],M2.shape()[2],R.shape()[1]]

    # anet = cytnx.Network()
    # anet.FromString(["psi: -1,-2;-3,-4",\
    #                  "L: -5;-1,0",\
    #                  "R: -7;-4,3",\
    #                  "M1: -5,-6;-2,1",\
    #                  "M2: -6,-7;-3,2",\
    #                  "TOUT: 0,1;2,3"])
    # #anet = cytnx.Network("projectorv2.net")
    # anet.PutUniTensor("M2",M2)
    # anet.PutUniTensors(["L","M1","R"],[L,M1,R])


    H = Hxx(anet,L,M1,M2,R)
    st = time.time()


    energy, psivec = cytnx.linalg.Lanczos(H, Tin=psivec, method = "Gnd", CvgCrit = 999, Maxiter = maxit)

    # psivec.set_labels(lbl)
    #energy ,psivec = cytnx.linalg.Lanczos_Gnd(H,CvgCrit=1.0e-12,Tin=psivec,maxiter=4000)

    en  = time.time()

    if(psivec.shape()==[chi, 2,2, chi]):
        Lanc_time.append(en-st)

    return psivec, energy.item()

print(cytnx.Device.Ncpus)


##### Set bond dimensions and simulation options
chi = 30
Nsites = 32
numsweeps = 15 # number of DMRG sweeps
maxit = 4 # iterations of Lanczos method
krydim = 4 # dimension of Krylov subspace


## Initialiaze MPO
##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
d = 2
s = 0.5

# bd_inner = cytnx.Bond(4,cytnx.BD_KET,[[0],[-2],[2],[0]]);
# bd_phys = cytnx.Bond(2,cytnx.BD_KET,[[1],[-1]]);
bd_inner = cytnx.Bond(cytnx.BD_KET,[[0],[-2],[2],[0]],[1,1,1,1])
bd_phys = cytnx.Bond(cytnx.BD_KET,[[1],[-1]],[1,1])

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


# q = 0 # conserving glb Qn
# VbdL = cytnx.Bond(1,cytnx.BD_KET,[[0]]);
# VbdR = cytnx.Bond(1,cytnx.BD_KET,[[q]]);
# L0 = cytnx.UniTensor([bd_inner.redirect(),VbdL.redirect(),VbdL],rowrank=1) #Left boundary
# R0 = cytnx.UniTensor([bd_inner,VbdR,VbdR.redirect()],rowrank=1) #Right boundary
# L0.set_elem([0,0,0],1); R0.set_elem([3,0,0],1);

q = 0 # conserving glb Qn
VbdL = cytnx.Bond(cytnx.BD_KET,[[0]],[1])
VbdR = cytnx.Bond(cytnx.BD_KET,[[q]],[1])
L0 = cytnx.UniTensor([bd_inner.redirect(),VbdL.redirect(),VbdL],rowrank=1) #Left boundary
R0 = cytnx.UniTensor([bd_inner,VbdR,VbdR.redirect()],rowrank=1) #Right boundary
L0.set_elem([0,0,0],1)
R0.set_elem([3,0,0],1)



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

A[0] = cytnx.UniTensor([VbdL,bd_phys.redirect(),cytnx.Bond(cytnx.BD_BRA,[[qcntr]],[1])],rowrank=2)
A[0].get_block_()[0] = 1
for k in range(1,Nsites):
    B1 = A[k-1].bonds()[2].redirect(); B2 = A[k-1].bonds()[1];

    if qcntr <= q:
        cq = 1
    else:
        cq = -1

    qcntr+=cq

    B3 = cytnx.Bond(cytnx.BD_BRA,[[qcntr]],[1])

    A[k] = cytnx.UniTensor([B1,B2,B3],rowrank=2)
    A[k].set_labels([2*k,2*k+1,2*k+2])

    A[k].get_block_()[0] = 1

    # print(A[k])


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
# anet.FromString(["L: -2;-1,-3",\
#                  "A: -1,-4;1",\
#                  "M: -2,0;-4,-5",\
#                  "A_Conj: 2;-3,-5",\
#                  "TOUT: 0;1,2"])
# anet.FromString(["L: a,b,c",\
#                  "A: b,d,a_",\
#                  "M: a,b_,d,e",\
#                  "A_Conj: c,e,c_",\
#                  "TOUT: a_;b_,c_"])
anet.FromString(["L: -2,-1,-3",\
                 "A: -1,-4,1",\
                 "M: -2,0,-4,-5",\
                 "A_Conj: -3,-5,2",\
                 "TOUT: 0;1,2"])
for p in range(Nsites - 1):
    anet.PutUniTensors(["L","A","A_Conj","M"],[LR[p],A[p],A[p].Dagger(),M])
    LR[p+1] = anet.Launch(optimal=True)



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

# exit()


## DMRG sweep
## Now we are ready for sweeping procedure!
##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Ekeep = []

for k in range(1, numsweeps+2):

    st_ = time.time()

    for p in range(Nsites-2,-1,-1):

        dim_l = A[p].shape()[0];
        dim_r = A[p+1].shape()[2];

        st = time.time()

        psi = cytnx.Contract(A[p],A[p+1]) ## contract
        en = time.time()
        if(psi.shape()==[chi, 2,2, chi]):
            AA_time.append(en-st)


        psi, Entemp = optimize_psi(psi, (LR[p],M,M,LR[p+2]), maxit, krydim)

        Ekeep.append(Entemp);
        #new_dim = min(dim_l*d,dim_r*d,chi)
        # print(psi.labels())
        lbl1 = A[p].labels()
        lbl2 = A[p+1].labels()

        psi.set_rowrank(2);

        s,A[p],A[p+1] = cytnx.linalg.Svd_truncate(psi, keepdim = chi)

        # s_,U,_ = cytnx.linalg.Svd(psi)
        # print("shape before truncate : ", s_.shape())

        A[p+1].set_labels(lbl2)
        # nsq = 0
        # bks = s.get_blocks()
        # for b in bks:
        #     for n in range(b.shape()[0]):
        #         val = b[n].item()
        #         nsq += val**2
        # for b in bks:
        #     for n in range(b.shape()[0]):
        #         b[n] = b[n].item()/(nsq**0.5)
        # for b in s.get_blocks():
        #     print(b.shape())
        # print("======================")
        # for b in s_.get_blocks():
        #     print(b.shape())
        # for b in A[p].get_blocks():
        #     print(b.shape())
        # print("======================")
        # for b in U.get_blocks():
        #     print(b.shape())
        # print(s.shape())
        A[p] = cytnx.Contract(A[p],s) ## absorb s into next neighbor

        A[p].set_labels(lbl1)

        # update LR from right to left:
        #anet = cytnx.Network("R_AMAH.net")
        # anet = cytnx.Network()
        # anet.FromString(["R: -2;-1,-3",\
        #                  "B: 1;-4,-1",\
        #                  "M: 0,-2;-4,-5",\
        #                  "B_Conj: -5,-3;2",\
        #                  "TOUT: 0;1,2"])
        # anet.FromString(["R: a,b,c",\
        #                  "B: b_,d,b",\
        #                  "M: a_,a,d,e",\
        #                  "B_Conj: c_,e,c",\
        #                  "TOUT: a_;b_,c_"])


        anet.FromString(["R: -2,-1,-3",\
                         "B: 1,-4,-1",\
                         "M: 0,-2,-4,-5",\
                         "B_Conj: 2,-5,-3",\
                         "TOUT: 0;1,2"])
        anet.PutUniTensors(["R","B","M","B_Conj"],[LR[p+2],A[p+1],M,A[p+1].Dagger()])
        LR[p+1] = anet.Launch(optimal=True)

        print('Sweep[r->l]: %d/%d, Loc:%d, Energy: %f'%(k,numsweeps,p,Ekeep[-1]))

    en_ = time.time()
    sweep_time.append(en_-st_)

    lbl = A[0].labels()
    A[0].set_rowrank(1)
    _,A[0] = cytnx.linalg.Svd(A[0],is_U=False, is_vT=True)
    A[0].set_labels(lbl)

    for p in range(Nsites-1):

        dim_l = A[p].shape()[0]
        dim_r = A[p+1].shape()[2]

        psi = cytnx.Contract(A[p],A[p+1]) ## contract


        psi, Entemp = optimize_psi(psi, (LR[p],M,M,LR[p+2]), maxit, krydim)
        Ekeep.append(Entemp)

        lbl1 = A[p].labels()
        lbl2 = A[p+1].labels()

        psi.set_rowrank(2)

        s,A[p],A[p+1] = cytnx.linalg.Svd_truncate(psi,keepdim = chi)
        s_,U,V = cytnx.linalg.Svd(psi)

        A[p].set_labels(lbl1)

        # nsq = 0
        # bks = s.get_blocks()
        # for b in bks:
        #     for n in range(b.shape()[0]):
        #         val = b[n].item()
        #         nsq += val**2
        # s /= nsq**0.5
        # print("==========ssss============")
        # for b in s.get_blocks():
        #     print(b.shape())
        # print("======================")
        # for b in s_.get_blocks():
        #     print(b.shape())

        # print("==========A[p+1]============")
        # for b in A[p+1].get_blocks():
        #     print(b.shape())
        # print("======================")
        # for b in V.get_blocks():
        #     print(b.shape())

        A[p+1] = cytnx.Contract(s,A[p+1]) ## absorb s into next neighbor.

        A[p+1].set_labels(lbl2)

        #anet = cytnx.Network("L_AMAH.net")
        anet = cytnx.Network()
        # anet.FromString(["L: -2;-1,-3",\
        #                  "A: -1,-4;1",\
        #                  "M: -2,0;-4,-5",\
        #                  "A_Conj: 2;-3,-5",\
        #                  "TOUT: 0;1,2"])
        anet.FromString(["L: -2,-1,-3",\
                        "A: -1,-4,1",\
                        "M: -2,0,-4,-5",\
                        "A_Conj: -3,-5,2",\
                        "TOUT: 0;1,2"])
        anet.PutUniTensors(["L","A","A_Conj","M"],[LR[p],A[p],A[p].Dagger(),M]);
        LR[p+1] = anet.Launch(optimal=True);


        print('Sweep[l->r]: %d/%d, Loc: %d, Energy: %f' % (k, numsweeps, p, Ekeep[-1]))


    lbl = A[-1].labels()
    A[-1].set_rowrank(2)
    _,A[-1] = cytnx.linalg.Svd(A[-1],is_U=True,is_vT=False) ## last one.
    A[-1].set_labels(lbl)
    print('done : %d'% k)



#### Compare with exact results (computed from free fermions)
from numpy import linalg as LA
# import matplotlib.pyplot as plt
H = np.diag(np.ones(Nsites-1),k=1) + np.diag(np.ones(Nsites-1),k=-1)
D = LA.eigvalsh(H)
EnExact = 2*sum(D[D < 0])
print("Exact : ")
print(EnExact)
print("lanczos time : ")
print(np.asarray(Lanc_time))
print("HAA time : ")
print(np.asarray(HAA_time))
print("AA time : ")
print(np.asarray(AA_time))
print("sweep time : ")
print(np.asarray(sweep_time))
exit()


##### Plot results
# plt.figure(1)
# plt.yscale('log')
# plt.plot(range(len(Ekeep)), np.array(Ekeep) - EnExact, 'b', label="chi = %d"%(chi), marker = 'o')
# plt.legend()
# plt.title('DMRG for XX model')
# plt.xlabel('Update Step')
# plt.ylabel('Ground Energy Error')
# plt.show()
