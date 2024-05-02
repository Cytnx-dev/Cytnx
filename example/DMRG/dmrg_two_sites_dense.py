import sys,os
import numpy as np
import cytnx

"""
Reference: https://www.tensors.net
Author: Yu-Hsueh Chen, Kai-Hsin Wu, Ke Hsu (j9263178)
"""

def dmrg_XXmodel_dense(Nsites, chi, numsweeps, maxit):
    class Hxx(cytnx.LinOp):

        def __init__(self, anet, psidim):
            cytnx.LinOp.__init__(self,"mv", psidim, cytnx.Type.Double, cytnx.Device.cpu)
            self.anet = anet

        def matvec(self, v):
            lbl = v.labels()
            self.anet.PutUniTensor("psi",v)
            out = self.anet.Launch()
            out.relabels_(lbl)
            return out

    def optimize_psi(psi, functArgs, maxit=2, krydim=4):

        L,M1,M2,R = functArgs
        anet = cytnx.Network()
        anet.FromString(["psi: -1,-2,-3,-4",\
                        "L: -5,-1,0",\
                        "R: -7,-4,3",\
                        "M1: -5,-6,-2,1",\
                        "M2: -6,-7,-3,2",\
                        "TOUT: 0,1;2,3"])
        # or you can do : anet = cytnx.Network("projector.net")
        anet.PutUniTensors(["L","M1","M2","R"],[L,M1,M2,R])

        H = Hxx(anet, psi.shape()[0]*psi.shape()[1]*psi.shape()[2]*psi.shape()[3])
        energy, psivec = cytnx.linalg.Lanczos(Hop = H, method = "Gnd", Maxiter = 4, CvgCrit = 9999999999, Tin = psi)

        return psivec, energy[0].item()

    d = 2 #physical dimension
    s = 0.5 #spin-half

    sx = cytnx.physics.spin(0.5,'x')
    sy = cytnx.physics.spin(0.5,'y')
    sp = sx+1j*sy
    sm = sx-1j*sy

    eye = cytnx.eye(d)
    M = cytnx.zeros([4, 4, d, d])
    M[0,0] = M[3,3] = eye
    M[0,1] = M[2,3] = 2**0.5*sp.real()
    M[0,2] = M[1,3] = 2**0.5*sm.real()
    M = cytnx.UniTensor(M,0)

    L0 = cytnx.UniTensor(cytnx.zeros([4,1,1]), rowrank = 0) #Left boundary
    R0 = cytnx.UniTensor(cytnx.zeros([4,1,1]), rowrank = 0) #Right boundary
    L0[0,0,0] = 1.; R0[3,0,0] = 1.

    lbls = [] # List for storing the MPS labels
    A = [None for i in range(Nsites)]
    A[0] = cytnx.UniTensor(cytnx.random.normal([1, d, min(chi, d)], 0., 1.), rowrank = 2)
    A[0].relabels_(["0","1","2"])
    lbls.append(["0","1","2"]) # store the labels for later convinience.

    for k in range(1,Nsites):
        dim1 = A[k-1].shape()[2]; dim2 = d
        dim3 = min(min(chi, A[k-1].shape()[2] * d), d ** (Nsites - k - 1))
        A[k] = cytnx.UniTensor(cytnx.random.normal([dim1, dim2, dim3],0.,1.), rowrank = 2)

        lbl = [str(2*k),str(2*k+1),str(2*k+2)]
        A[k].relabels_(lbl)
        lbls.append(lbl) # store the labels for later convinience.

    LR = [None for i in range(Nsites+1)]
    LR[0]  = L0
    LR[-1] = R0


    for p in range(Nsites - 1):

        ## Changing to canonical form site by site:
        s, A[p] ,vt = cytnx.linalg.Gesvd(A[p])
        A[p+1] = cytnx.Contract(cytnx.Contract(s,vt),A[p+1])

        ## Calculate enviroments:
        anet = cytnx.Network()
        anet.FromString(["L: -2,-1,-3",\
                        "A: -1,-4,1",\
                        "M: -2,0,-4,-5",\
                        "A_Conj: -3,-5,2",\
                        "TOUT: 0,1,2"])
        # or you can do: anet = cytnx.Network("L_AMAH.net")
        anet.PutUniTensors(["L","A","A_Conj","M"],[LR[p],A[p],A[p].Conj(),M])
        LR[p+1] = anet.Launch()

        # Recover the original MPS labels
        A[p].relabels_(lbls[p])
        A[p+1].relabels_(lbls[p+1])

    _,A[-1] = cytnx.linalg.Gesvd(A[-1],is_U=True,is_vT=False) ## last one.
    A[-1].relabels_(lbls[-1]) # Recover the original MPS labels

    Ekeep = []
    for k in range(1, numsweeps+1):

        for p in range(Nsites-2,-1,-1):
            dim_l = A[p].shape()[0]
            dim_r = A[p+1].shape()[2]
            new_dim = min(dim_l*d,dim_r*d,chi)

            psi = cytnx.Contract(A[p],A[p+1]) # contract
            psi, Entemp = optimize_psi(psi, (LR[p],M,M,LR[p+2]), maxit)
            Ekeep.append(Entemp)

            psi.set_rowrank_(2) # maintain rowrank to perform the svd
            s,A[p],A[p+1] = cytnx.linalg.Svd_truncate(psi,new_dim)
            A[p+1].relabels_(lbls[p+1]); # set the label back to be consistent

            s = s/s.Norm().item() # normalize s

            A[p] = cytnx.Contract(A[p],s) # absorb s into next neighbor
            A[p].relabels_(lbls[p]); # set the label back to be consistent

            # update LR from right to left:
            anet = cytnx.Network()
            anet.FromString(["R: -2,-1,-3",\
                            "B: 1,-4,-1",\
                            "M: 0,-2,-4,-5",\
                            "B_Conj: 2,-5,-3",\
                            "TOUT: 0;1,2"])
            # or you can do: anet = cytnx.Network("R_AMAH.net")
            anet.PutUniTensors(["R","B","M","B_Conj"],[LR[p+2],A[p+1],M,A[p+1].Conj()])
            LR[p+1] = anet.Launch()

            print('Sweep[r->l]: %d/%d, Loc: %d,Energy: %f' % (k, numsweeps, p, Ekeep[-1]))

        A[0].set_rowrank_(1)
        _,A[0] = cytnx.linalg.Gesvd(A[0],is_U=False, is_vT=True)
        A[0].relabels_(lbls[0]); #set the label back to be consistent

        for p in range(Nsites-1):
            dim_l = A[p].shape()[0]
            dim_r = A[p+1].shape()[2]
            new_dim = min(dim_l*d,dim_r*d,chi)

            psi = cytnx.Contract(A[p],A[p+1]) ## contract
            psi, Entemp = optimize_psi(psi, (LR[p],M,M,LR[p+2]), maxit)
            Ekeep.append(Entemp)

            psi.set_rowrank_(2) # maintain rowrank to perform the svd
            s,A[p],A[p+1] = cytnx.linalg.Svd_truncate(psi,new_dim)
            A[p].relabels_(lbls[p]); #set the label back to be consistent

            s = s/s.Norm().item() # normalize s

            A[p+1] = cytnx.Contract(s,A[p+1]) ## absorb s into next neighbor.
            A[p+1].relabels_(lbls[p+1]); #set the label back to be consistent

            # update LR from left to right:
            anet = cytnx.Network()
            anet.FromString(["L: -2,-1,-3",\
                            "A: -1,-4,1",\
                            "M: -2,0,-4,-5",\
                            "A_Conj: -3,-5,2",\
                            "TOUT: 0,1,2"])
            # or you can do: anet = cytnx.Network("L_AMAH.net")

            anet.PutUniTensors(["L","A","A_Conj","M"],[LR[p],A[p],A[p].Conj(),M])
            LR[p+1] = anet.Launch()

            print('Sweep[l->r]: %d/%d, Loc: %d,Energy: %f' % (k, numsweeps, p, Ekeep[-1]))

        A[-1].set_rowrank_(2)
        _,A[-1] = cytnx.linalg.Gesvd(A[-1],is_U=True,is_vT=False) ## last one.
        A[-1].relabels_(lbls[-1]); #set the label back to be consistent
    return Ekeep

if __name__ == '__main__':


    Nsites = 20 # Number of sites
    chi = 32 # MPS bond dimension
    numsweeps = 6 # number of DMRG sweeps
    maxit = 2 # iterations of Lanczos method

    Es = dmrg_XXmodel_dense(Nsites, chi, numsweeps, maxit)

    #### Compare with exact results (computed from free fermions)
    from numpy import linalg as LA
    # import matplotlib.pyplot as plt
    H = np.diag(np.ones(Nsites-1),k=1) + np.diag(np.ones(Nsites-1),k=-1)
    D = LA.eigvalsh(H)
    EnExact = 2*sum(D[D < 0])

    print("Energy error = ", np.abs(Es[-1]-EnExact))
