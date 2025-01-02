import sys,os
import numpy as np
import cytnx

"""
Author: Hao-Ti Hung
"""

def tdvp1_XXZmodel_dense(J, Jz, hx, hz, A, chi, dt, time_step):

    class OneSiteOp(cytnx.LinOp):
        def __init__(self, L, M, R):
            self.anet = cytnx.Network()
            d = L.shape()[0]
            D1 = L.shape()[2]
            D2 = R.shape()[2]
            cytnx.LinOp.__init__(self, "mv", D1*D2*d, L.dtype(), R.device())
            self.anet.FromString([\
                        "psi: -1,-2,-3",\
                        "L: -4,-1,0",\
                        "R: -6,-3,2",\
                        "M: -4,-6,-2,1",\
                        "TOUT: 0,1;2"])
            self.anet.PutUniTensors(["L","M","R"],[L,M,R])
        def matvec(self, v):
            self.anet.PutUniTensor("psi",v)
            out = self.anet.Launch()
            out.relabels_(v.labels())
            return out

    class ZeroSiteOp(cytnx.LinOp):
        def __init__(self, L, R):
            self.anet = cytnx.Network()
            d = L.shape()[0]
            D1 = L.shape()[2]
            D2 = R.shape()[2]
            cytnx.LinOp.__init__(self, "mv", D1*D2, L.dtype(), R.device())
            self.anet.FromString([\
                        "C: -1,-2",\
                        "L: -3,-1,0",\
                        "R: -3,-2,1",\
                        "TOUT: 0;1"])
            self.anet.PutUniTensors(["L","R"],[L,R])
        def matvec(self, v):
            self.anet.PutUniTensor("C",v)
            out = self.anet.Launch()
            out.relabels_(v.labels())
            return out

    def time_evolve_Lan_f(psi, functArgs, delta):
        L,M,R = functArgs
        L = L.astype(cytnx.Type.ComplexDouble)
        M = M.astype(cytnx.Type.ComplexDouble)
        R = R.astype(cytnx.Type.ComplexDouble)
        op = OneSiteOp(L,M,R)
        exp_iH_v = cytnx.linalg.Lanczos_Exp(op, psi, -1.0j*delta*0.5, 1.0e-8)
        exp_iH_v.relabels_(psi.labels())
        return exp_iH_v

    def time_evolve_Lan_b(psi, functArgs, delta):
        L,R = functArgs
        L = L.astype(cytnx.Type.ComplexDouble)
        R = R.astype(cytnx.Type.ComplexDouble)
        op = ZeroSiteOp(L,R)
        exp_iH_v = cytnx.linalg.Lanczos_Exp(op, psi, 1.0j*delta*0.5, 1.0e-8)
        exp_iH_v.relabels_(psi.labels())
        return exp_iH_v

    def get_energy(A, M):
        N = len(A)
        L0 = cytnx.UniTensor(cytnx.zeros([5,1,1]), rowrank = 0) #Left boundary
        R0 = cytnx.UniTensor(cytnx.zeros([5,1,1]), rowrank = 0) #Right boundary
        L0[0,0,0] = 1.; R0[4,0,0] = 1.
        L = L0
        anet = cytnx.Network()
        anet.FromString(["L: -2,-1,-3",\
                        "A: -1,-4,1",\
                        "M: -2,0,-4,-5",\
                        "A_Conj: -3,-5,2",\
                        "TOUT: 0,1,2"])
        # or you can do: anet = cytnx.Network("L_AMAH.net")
        for p in range(0, N):
            anet.PutUniTensors(["L","A","A_Conj","M"],[L,A[p],A[p].Conj(),M])
            L = anet.Launch()
        E = cytnx.Contract(L, R0).item()
        print('energy:', E)
        return E


    d = 2 #physical dimension
    sx = cytnx.physics.pauli('x').real()
    sy = cytnx.physics.pauli('y')
    sz = cytnx.physics.pauli('z').real()
    sp = (sx+1j*sy).real()
    sm = (sx-1j*sy).real()

    eye = cytnx.eye(d)
    M = cytnx.zeros([5, 5, d, d])
    M[0,0] = M[4,4] = eye
    M[0,4] = hx*sx + hz*sz
    M[1,4] = sp
    M[2,4] = sm
    M[3,4] = sz
    M[0,1] = 0.5*J*sm
    M[0,2] = 0.5*J*sp
    M[0,3] = Jz*sz
    M = cytnx.UniTensor(M,0)

    L0 = cytnx.UniTensor(cytnx.zeros([5,1,1]), rowrank = 0) #Left boundary
    R0 = cytnx.UniTensor(cytnx.zeros([5,1,1]), rowrank = 0) #Right boundary
    L0[0,0,0] = 1.; R0[4,0,0] = 1.

    lbls = [] # List for storing the MPS labels
    Nsites = len(A)
    lbls.append(["0","1","2"]) # store the labels for later convinience.

    for k in range(1,Nsites):
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

    As = []
    As.append(A.copy())
    Es = []


    for k in range(1, time_step+1):
        print('time:', k)
        E = get_energy(A, M)
        Es.append(E)

        for p in range(Nsites-1,-1,-1):
            dim_l = A[p].shape()[0]
            dim_r = A[p].shape()[2]
            new_dim = min(dim_l*d,dim_r*d,chi)


            psi = A[p].clone()
            psi = time_evolve_Lan_f(psi, (LR[p],M,LR[p+1]), dt)

            psi.set_rowrank_(1) # maintain rowrank to perform the svd
            s,_,A[p] = cytnx.linalg.Svd_truncate(psi,new_dim)
            A[p].relabels_(lbls[p]); # set the label back to be consistent
            # update LR from right to left:
            anet = cytnx.Network()
            anet.FromString(["R: -2,-1,-3",\
                            "B: 1,-4,-1",\
                            "M: 0,-2,-4,-5",\
                            "B_Conj: 2,-5,-3",\
                            "TOUT: ;0,1,2"])
            # or you can do: anet = cytnx.Network("R_AMAH.net")
            anet.PutUniTensors(["R","B","M","B_Conj"],[LR[p+1],A[p],M,A[p].Conj()])
            old_LR = LR[p].clone()
            if p != 0:
                LR[p] = anet.Launch()
                s = s/s.Norm().item() # normalize s
                C = cytnx.Contract(_, s)
                #C = time_evolve_b(C, (old_LR, LR[p]), dt)
                C = time_evolve_Lan_b(C, (old_LR, LR[p]), dt)

                A[p-1] = cytnx.Contract(A[p-1], C).relabels_(A[p-1].labels())




            print('Sweep[r->l]: %d/%d, Loc: %d' % (k, time_step, p))

        A[0].set_rowrank_(1)
        _,A[0] = cytnx.linalg.Gesvd(A[0],is_U=False, is_vT=True)
        A[0].relabels_(lbls[0]); #set the label back to be consistent


        for p in range(Nsites):
            dim_l = A[p].shape()[0]
            dim_r = A[p].shape()[2]
            new_dim = min(dim_l*d,dim_r*d,chi)

            psi = A[p]
            psi = time_evolve_Lan_f(psi, (LR[p],M,LR[p+1]), dt)

            psi.set_rowrank_(2) # maintain rowrank to perform the svd
            s,A[p],_ = cytnx.linalg.Svd_truncate(psi,new_dim)
            A[p].relabels_(lbls[p]); #set the label back to be consistent
            # update LR from left to right:
            anet = cytnx.Network()
            anet.FromString(["L: -2,-1,-3",\
                            "A: -1,-4,1",\
                            "M: -2,0,-4,-5",\
                            "A_Conj: -3,-5,2",\
                            "TOUT: 0,1,2"])

            anet.PutUniTensors(["L","A","A_Conj","M"],[LR[p],A[p],A[p].Conj(),M])
            old_LR = LR[p+1].clone()


            if p != Nsites - 1:
                LR[p+1] = anet.Launch()
                s = s/s.Norm().item() # normalize s
                C = cytnx.Contract(s, _)
                C = time_evolve_Lan_b(C, (LR[p+1],old_LR), dt)
                A[p+1] = cytnx.Contract(A[p+1], C)
                A[p+1].permute_(['_aux_L', lbls[p+1][1], lbls[p+1][2]])
                A[p+1].relabels_(lbls[p+1])

            print('Sweep[l->r]: %d/%d, Loc: %d' % (k, time_step, p))

        A[-1].set_rowrank_(2)
        _,A[-1] = cytnx.linalg.Gesvd(A[-1],is_U=True,is_vT=False) ## last one.
        A[-1].relabels_(lbls[-1]); #set the label back to be consistent
        As.append(A.copy())
    return As, Es # all time step states

def Local_meas(A, B, Op, site):
    N = len(A)
    l = cytnx.UniTensor(cytnx.eye(1), rowrank = 1)
    anet = cytnx.Network()
    anet.FromString(["l: 0,3",\
                    "A: 0,1,2",\
                    "B: 3,1,4",\
                    "TOUT: 2;4"])
    for i in range(0, N):
        if i != site:
            anet.PutUniTensors(["l","A","B"],[l,A[i],B[i].Conj()])
            l = anet.Launch()
        else:
            tmp = A[i].relabel(1, "_aux_up")
            Op = Op.relabels(["_aux_up", "_aux_low"])
            tmp = cytnx.Contract(tmp, Op)
            tmp.relabel_("_aux_low", A[i].labels()[1])
            tmp.permute_(A[i].labels())
            anet.PutUniTensors(["l","A","B"],[l,tmp,B[i].Conj()])
            l = anet.Launch()

    return l.reshape(1).item()


def prepare_rand_init_MPS(Nsites, chi, d):
    lbls = []
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
    return A


if __name__ == '__main__':
    #prepare random MPS
    Nsites = 7 # Number of sites
    chi = 8 # MPS bond dimension
    d = 2
    MPS_rand = prepare_rand_init_MPS(Nsites, chi, d)

    # simulate ground state by imaginary time evolution by tdvp
    J = 0.0
    Jz = 0.0
    hx = 0.0
    hz = -1.0
    tau = -1.0j
    time_step = 10
    # prepare up state
    As, Es = tdvp1_XXZmodel_dense(J, Jz, hx, hz, MPS_rand, chi, tau, time_step)
    GS = As[time_step - 1]

    # real tiem evoolution
    J = 0.0
    Jz = 0.5
    hx = 0.3
    hz = 3.0
    dt = 0.1
    time_step = 25 # number of DMRG sweeps
    As, Es = tdvp1_XXZmodel_dense(J, Jz, hx, hz, GS, chi, dt, time_step)

    # measure middle site <Sz>
    Sz = cytnx.UniTensor(cytnx.physics.pauli('z').real(), rowrank = 1)
    Szs = []
    mid_site = int(Nsites/2)
    for i in range(0, len(As)):
        Szs.append(Local_meas(As[i], As[i], Sz, mid_site).real)
