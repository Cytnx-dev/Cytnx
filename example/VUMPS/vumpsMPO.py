import time
import numpy as np
import cytnx as cy
from ncon import ncon
from numpy import linalg
from scipy.sparse.linalg import LinearOperator, eigsh, eigs, bicgstab

'''
1. Need Arnoldi
2. Need bicgstab or gmres (LW and RW are calculated in numpy)
3. Lanczos_ER much slower than scipy.eigsh, Lanczos_Gnd doesn't work.
'''
    
def tonumpy(uniten):
    return uniten.get_block_().numpy()

def vumpsMPO(W, AL, AR, C, LW = None, RW = None, eigsolve_tol = 1e-12, grad_tol = 1e-8, energy_tol = 1e-8, maxit = 50):
    
    D = AL.shape()[0] # bond dimension
    d = AL.shape()[1] # physical dimension
    dw = W.shape()[0]
    W_ = tonumpy(W) # W_ for getLW and getRW

    def getLW(LW, AL):

        def getR(AL):
            TL = ncon([AL, AL.conj()], [[-1,1,-3], [-2,1,-4]]).reshape(D**2, D**2)
            # TL_ = np.transpose(TL, (0, 2, 1, 3))
            R = eigs(TL, k = 1, which = 'LM')[1].reshape(D, D)
            R = 0.5*(R+R.conj().T)
            R/=ncon([R,np.eye(D)],[[1,2],[1,2]])
            return np.real(R)

        def getLWaC21(LWa, YLa, TL, la):
            m = LWa.shape[0]
            def Lcontract(LWa):
                m = TL.shape[0]
                b = TL.shape[1]
                LWa_TL = ncon([LWa, TL],[[1,2],[1,-1,2,-2]])
                return LWa.flatten() - la*LWa_TL.flatten()

            LeftOp = LinearOperator((m**2, m**2), matvec=Lcontract, dtype=np.float64)
            LWa_temp, is_conv = gmres(LeftOp, YLa.flatten(), x0=LWa.flatten(), tol=1e-10,
                                    restart=None, atol=None)
            LWa_temp = LWa_temp.reshape(m, m)
            # La = 0.5 * (La_temp + La_temp.T) investgate
            return LWa_temp
    
        def getLWaC25(LWa, YLa, AL, R):
            m = AL.shape[0]
            def Lcontract(v):
                v = v.reshape(m,m)
                LWa_TL = ncon([v, AL, AL.conj()],[[1,2],[1,3,-1],[2,3,-2]])
                LWa_P = ncon([v, R], [[1,2],[1,2]])*np.eye(m)
                return v.flatten() - LWa_TL.flatten() + LWa_P.flatten()
            LeftOp = LinearOperator((m**2, m**2), matvec=Lcontract, dtype=np.float64)
            B = YLa - ncon([YLa, R], [[1,2],[1,2]])*np.eye(m)
            LWa_temp, is_conv = bicgstab(LeftOp, B.flatten(), x0=LWa.flatten(), tol=1e-12)
            LWa_temp = np.real(LWa_temp).reshape(m, m)
            return LWa_temp

        YL = np.zeros([dw, D, D])
        LW_ = LW.copy()
        LW_[dw-1] = np.eye(D)
        R = getR(AL)
        for a in range(dw-2, -1, -1):
            for b in range(a+1, dw):
                YL[a] += ncon([LW_[b], AL, W_[b,a], AL.conj()],[[1,2], [1,4,-1], [4,5], [2,5,-2]])
            if W_[a, a, 0, 0] == 0:
                LW_[a] = YL[a]
            elif W_[a, a, 0, 0] == 1:
                LW_[a] = getLWaC25(LW_[a], YL[a], AL, R)
            else:  #W[a,a] = constant*I
                LW_[a] = getLWaC21(LW_[a], YL[a], AL, W_[a, a, 0, 0])

        return cy.UniTensor(cy.from_numpy(np.asarray(LW_)), 0), ncon([YL[0], R], [[1,2],[1,2]])

    def getRW(RW, AR):

        def getL(AR):
            TR = ncon([AR, AR.conj()], [[-3,1,-1], [-4,1,-2]]).reshape(D**2, D**2)
            # TR_ = np.transpose(TR, (0, 2, 1, 3))
            # TR = TR.T
            L = eigs(TR, k = 1, which = 'LM')[1].reshape(D, D)
            L = 0.5*(L+L.conj().T)
            L /= ncon([L,np.eye(D)],[[1,2],[1,2]])
            return np.real(L)

        def getRWaC21(RWa, YRa, TR, la):
            m = RWa.shape[0]
            def Rcontract(RWa):
                m = TR.shape[0]
                b = TR.shape[1]
                RWa_TR = ncon([RWa, TR],[[1,2],[-1,1,-2,2]])
                return RWa.flatten() - la*RWa_TR.flatten()

            RightOp = LinearOperator((m**2, m**2), matvec=Rcontract, dtype=np.float64)
            RWa_temp, is_conv = gmres(RightOp, YRa.flatten(), x0=RWa.flatten(), tol=1e-10,
                                    restart=None, atol=None)
            RWa_temp = RWa_temp.reshape(m, m)
            # La = 0.5 * (La_temp + La_temp.T) investgate
            return RWa_temp

        def getRWaC25(RWa, YRa, AR, L):
            m = AR.shape[0]
            def Rcontract(v):
                v = v.reshape(m,m)
                RWa_TR = ncon([v, AR, AR.conj()],[[1,2], [-1,3,1], [-2,3,2]])
                RWa_P = ncon([L, v], [[1,2],[1,2]])*np.eye(m)
                return v.flatten() - RWa_TR.flatten() + RWa_P.flatten()
            RightOp = LinearOperator((m**2, m**2), matvec=Rcontract, dtype=np.float64)
            B = YRa - ncon([L, YRa], [[1,2],[1,2]])*np.eye(m)
            RWa_temp, is_conv = bicgstab(RightOp, B.flatten(), x0=RWa.flatten(), tol=1e-12)
            RWa_temp = np.real(RWa_temp).reshape(m, m)
            return RWa_temp

        YR = np.zeros([dw, D, D])
        RW_ = RW.copy()
        RW_[0] = np.eye(D)
        L = getL(AR)

        for a in range(1, dw):
            for b in range(a-1, -1, -1):
                YR[a] += ncon([RW_[b], AR, W_[a,b], AR.conj()],[[1,2], [-1,4,1], [4,5], [-2,5,2]])
            if W_[a, a, 0, 0] == 0:
                RW_[a] = YR[a]
            elif W_[a, a, 0, 0] == 1:
                RW_[a] = getRWaC25(RW_[a], YR[a], AR, L)
            else:  #W[a,a] = a constant*I
                RW_[a] = getRWaC21(RW_[a], YR[a], AR, W_[a, a, 0, 0])

        return cy.UniTensor(cy.from_numpy(np.asarray(RW_)), 0), ncon([L, YR[-1]], [[1,2],[1,2]])

    def applyH_AC(LW, W, RW, AC):

        LW, W, RW, AC = tonumpy(LW), tonumpy(W), tonumpy(RW), tonumpy(AC)
        def MidTensor(AC_mat):
            AC_mat=AC_mat.reshape([D, d, D])
            tensors = [LW, W, RW, AC_mat]
            labels = [[2,1,-1], [2,3,4,-2], [3,5,-3], [1,4,5]]
            return (ncon(tensors, labels)).flatten()

        TensorOp = LinearOperator((d * D**2, d * D**2),
                                matvec=MidTensor, dtype=np.float64)

        AC_new = eigsh(TensorOp, k=1, which='SA', v0=AC.flatten(),
                    ncv=None, maxiter=None, tol=eigsolve_tol)[1].reshape(D, d, D)
        return cy.UniTensor(cy.from_numpy(AC_new), 0)

        # class H_AC(cy.LinOp):
        #     def __init__(self):
        #         cy.LinOp.__init__(self,"mv", D*D*d, cy.Type.Double, cy.Device.cpu)
        #         self.net = cy.Network()
        #         self.net.FromString(["AC: ;-1,-2,-3", "LW: ;-4,-1,0", "RW: ;-5,-3,2", "W: ;-4,-5,-2,1", "TOUT: ;0,1,2"])
        #         self.net.PutUniTensors(["LW","W","RW"],[LW, W, RW], False)

        #     def matvec(self, v):
        #         vu = cy.UniTensor(v.reshape(D, d, D), 0) ## share memory, no copy
        #         self.net.PutUniTensor("AC", vu, False)
        #         out = self.net.Launch(optimal=True).get_block_() # get_block_ without copy
        #         out.flatten_() # only change meta, without copy.
        #         return out
        # # e, AC_new = cy.linalg.Lanczos_Gnd(H_AC(), CvgCrit = 1.0e-8, Tin = AC.get_block_().flatten(), maxiter = 10000)
        # e, AC_new = cy.linalg.Lanczos_ER(H_AC(), k = 1, maxiter = 10000, CvgCrit = 1e-8, Tin = AC.get_block_().flatten(), max_krydim = 3)
        # return cy.UniTensor(AC_new.reshape(D, d, D), 0)

    def applyH_C(LW, RW, C):
        # Numpy version
        LW, RW, C = tonumpy(LW), tonumpy(RW), tonumpy(C) 
        def MidWeights(C_mat):
            C_mat = C_mat.reshape(D, D)
            tensors = [LW, C_mat, RW]
            labels = [[2,1,-1], [1,3], [2,3,-2]]
            con_order = [1,3,2]
            return (ncon(tensors, labels)).flatten()
        WeightOp = LinearOperator((D**2, D**2), matvec=MidWeights, dtype=np.float64)
        C_temp = eigsh(WeightOp, k=1, which='SA', v0=C.flatten(),
                    ncv=None, maxiter=None, tol=eigsolve_tol)[1]

        C_temp = cy.UniTensor(cy.from_numpy(C_temp.reshape(D, D)), 1)
        C_new, u, vt = cy.linalg.Svd(C_temp)
        C_new = cy.UniTensor(cy.linalg.Diag(C_new.get_block_()), 0)
        return u, C_new, vt

        ## Uniten version for Lanczos_Gnd_Ut
        # class H_C(cy.LinOp):
        #     def __init__(self):
        #         cy.LinOp.__init__(self,"mv", D*D, cy.Type.Double, cy.Device.cpu)
        #         self.net = cy.Network()
        #         self.net.FromString(["LW: ;-1,-2,0", "C: -2;-3", "RW: ;-1,-3,1", "TOUT: 0;1"])
        #         self.net.PutUniTensors(["LW", "RW"],[LW, RW], False)

        #     def matvec(self, v):
        #         lbl = v.labels()
        #         self.net.PutUniTensor("C", v)
        #         out = self.net.Launch(optimal=True)
        #         out.set_labels(lbl)
        #         out.contiguous_()
        #         return out

        # e, C_temp = cy.linalg.Lanczos_Gnd_Ut(H_C(), CvgCrit = 1e-12, is_V = True, Tin = C, verbose = False, maxiter = 100000)
        # C_new, u, vt = cy.linalg.Svd(C_temp)
        # C_new = cy.UniTensor(cy.linalg.Diag(C_new.get_block_()), 1)
        # return u, C_new, vt

        ## vector version for Lanczos_Gnd or Lanczos_GR
        # class H_C(cy.LinOp):
        #     def __init__(self):
        #         cy.LinOp.__init__(self,"mv", D*D, cy.Type.Double, cy.Device.cpu)
        #         self.net = cy.Network()
        #         self.net.FromString(["LW: ;-1,-2,0", "C: -2;-3", "RW: ;-1,-3,1", "TOUT: 0;1"])
        #         self.net.PutUniTensors(["LW", "RW"],[LW, RW], False)

        #     def matvec(self, v):
        #         vu = cy.UniTensor(v.reshape(D, D), 1) ## share memory, no copy
        #         self.net.PutUniTensor("C", vu, False)
        #         out = self.net.Launch(optimal=True).get_block_() # get_block_ without copy
        #         out.flatten_() # only change meta, without copy.
        #         return out
        # # e, C_temp = cy.linalg.Lanczos_Gnd(H_C(), CvgCrit = 1.0e-8, Tin = C.get_block_().flatten(), maxiter = 10000)
        # e, C_temp = cy.linalg.Lanczos_ER(H_C(), k = 1, maxiter = 10000, CvgCrit = 1e-8, Tin = C.get_block_().flatten(), max_krydim = 3)
        # C_new, u, vt = cy.linalg.Svd(cy.UniTensor(C_temp.reshape(D, D), 1))
        # C_new = cy.UniTensor(cy.linalg.Diag(C_new.get_block_()), 1)
        # return u, C_new, vt

    def updateALAR(AL, AR, AC):

        # AL = (polar(AC.reshape(D * d, D), side='right')[0]).reshape(D, d, D)
        # AR = (polar(AC.reshape(D, d * D), side='left')[0]).reshape(D, d, D)

        # ut, _, vt = linalg.svd(AC.reshape(D * d, D), full_matrices=False)
        # AL = (ut @ vt).reshape(D, d, D)
        # ut, _, vt = linalg.svd(AC.reshape(D, d * D), full_matrices=False)
        # AR = (ut @ vt).reshape(D, d, D)
        # return cy.UniTensor(cy.from_numpy(AL), 0), cy.UniTensor(cy.from_numpy(AR), 0)

        AC.set_rowrank(2)
        s, u, vt = cy.linalg.Svd(AC)
        u.set_labels([-1,-2,1]); vt.set_labels([1,-3])
        AL = cy.Contract(u, vt).reshape(D, d, D)
        AC.set_rowrank(1)

        s, u, vt = cy.linalg.Svd(AC)
        u.set_labels([-1,1]); vt.set_labels([1,-2,-3])
        AR = cy.Contract(u, vt).reshape(D, d, D)

        return AL, AR


    tAL = AL.relabels([0, 1, 2]); tC = C.relabels([2, 3])
    AC = cy.Contract(tAL, tC)

    if LW is None:
        LW = cy.UniTensor(cy.random.normal([dw, D, D],0.,1.), 0)
    if RW is None:
        RW = cy.UniTensor(cy.random.normal([dw, D, D],0.,1.), 0)

    # el, er = 9999, 9999
    Energy, Energynew = 0, 0

    ucon = cy.Network()
    ucon.FromString(["ut: 0;-1",\
                    "AL: ;-1,1,-2",\
                    "u: -2;2",\
                    "TOUT: ;0,1,2"])

    for k in range(maxit): 

        time_start = time.time()

        LW, EnergyL = getLW(tonumpy(LW), tonumpy(AL))
        RW, EnergyR = getRW(tonumpy(RW), tonumpy(AR))

        Energynew = (EnergyL+EnergyR)/2
        print("iteration %d, energy = %.10f"%(k, Energynew))
        if abs(Energynew-Energy)<energy_tol:
            print("Tolerence achieved!")
            break
        Energy = Energynew

        u, C, vt = applyH_C(LW, RW, C)

        ucon.PutUniTensors(["ut","AL","u"], [u.Transpose(), AL, u])
        AL = ucon.Launch(optimal=True)
        ucon.PutUniTensors(["ut","AL","u"], [vt, AR, vt.Transpose()])
        AR = ucon.Launch(optimal=True)

        LW, _ = getLW(tonumpy(LW), tonumpy(AL))
        RW, _ = getRW(tonumpy(RW), tonumpy(AR))

        AC = applyH_AC(LW, W, RW, AC)
        AL, AR = updateALAR(AL, AR, AC)

        time_end = time.time()
        # print("total time: "+str(time_end - time_start))

    return AL, C, AR, LW, RW, Energy

if __name__ == '__main__':
    D = 64
    d = 2
    sx = cy.physics.spin(0.5,'x')
    sy = cy.physics.spin(0.5,'y')
    sp = sx+1j*sy
    sm = sx-1j*sy
    eye = cy.eye(d)
    W = cy.zeros([4, 4, d, d])
    W[0,0] = W[3,3] = eye
    W[0,1] = W[2,3] = 2**0.5*sp.real()
    W[0,2] = W[1,3] = 2**0.5*sm.real()
    W = cy.UniTensor(W, 0)
    W.permute_([1,0,2,3]) # In the paper the MPO is of lower tridiagonal

    C = cy.UniTensor(cy.linalg.Diag(cy.random.normal([D], 0., 1.)), 1)
    C =  C / C.get_block_().Norm().item()
    AL = (cy.linalg.Svd(cy.UniTensor(cy.random.normal([D*d, D],0.,1.), 1))[1]).reshape(D, d, D)
    AR = (cy.linalg.Svd(cy.UniTensor(cy.random.normal([D, D*d],0.,1.), 1))[2]).reshape(D, d, D)

    vumpsMPO(W, AL, AR, C)

    