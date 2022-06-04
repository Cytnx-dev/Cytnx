import numpy as np
import cytnx

"""
Reference: https://arxiv.org/abs/0804.2509v1
Author: Kai-Hsin Wu 
"""


class Projector(cytnx.LinOp):
   
    

    def __init__(self,L,M1,M2,R,psi_dim,psi_dtype,psi_device):
        cytnx.LinOp.__init__(self,"mv",psi_dim,psi_dtype,psi_device)
        
        self.anet = cytnx.Network("projector.net")
        self.anet.PutUniTensor("M2",M2)
        self.anet.PutUniTensors(["L","M1","R"],[L,M1,R],False)
        self.psi_shape = [L.shape()[1],M1.shape()[2],M2.shape()[2],R.shape()[1]]      
  
    def matvec(self,psi):
        
        psi_p = cytnx.UniTensor(psi.clone(),rowrank=0)  ## clone here
        psi_p.reshape_(*self.psi_shape)

        self.anet.PutUniTensor("psi",psi_p,False) ## no- redundant clone here
        H_psi = self.anet.Launch(optimal=True).get_block_() # get_block_ without copy

        H_psi.flatten_()
        return H_psi



def eig_Lanczos(psivec, functArgs, Cvgcrit=1.0e-15,maxit=100000):
    """ Lanczos method for finding smallest algebraic eigenvector of linear \
    operator defined as a function"""
    #print(eig_Lanczos)

    Hop = Projector(*functArgs,psivec.shape()[0],psivec.dtype(),psivec.device())
    gs_energy ,psivec = cytnx.linalg.Lanczos_Gnd(Hop,Cvgcrit,Tin=psivec,maxiter=maxit)

    return psivec, gs_energy.item()
    


##### Set bond dimensions and simulation options
chi = 64;
Niter = 100; # number of iteration of DMRG
maxit = 100000 # iterations of Lanczos method

## Initialiaze MPO 
##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
## Here we consider Hamiltonian with 
#
#    [J]*SzSz + 2[Hx]*Sx
#
#
J  = 1.0
Hx = 1.0

d = 2
sx = cytnx.physics.pauli('x').real()
sz = cytnx.physics.pauli('z').real()
eye = cytnx.eye(d)
M = cytnx.zeros([3, 3, d, d])
M[0,0] = M[2,2] = eye
M[0,1] = M[1,2] = sz
M[0,2] = 2*Hx*sx
M = cytnx.UniTensor(M,rowrank=0)

L0 = cytnx.UniTensor(cytnx.zeros([3,1,1]),rowrank=0) #Left boundary
R0 = cytnx.UniTensor(cytnx.zeros([3,1,1]),rowrank=0) #Right boundary
L0.get_block_()[0,0,0] = 1.; R0.get_block_()[2,0,0] = 1.;

## Local Measurement Operator:
## Here, we consider a local measurement of energy.
H = J*cytnx.linalg.Kron(sz,sz) + Hx*(cytnx.linalg.Kron(sx,eye) + cytnx.linalg.Kron(eye,sx))
H = cytnx.UniTensor(H.reshape(2,2,2,2),rowrank=2)

## Init the left and right enviroment
#
#   L[0]:       R[0]:
#       ---         ---
#       |             |
#      L0--         --R0
#       |             |
#       ---         ---
#
##>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
L = L0
R = R0


## Construct n = 0
#  1) init a random psi as initial state with shape [1,d,d,1]
#
#      (1)---[psi0]---(1)
#            |    |
#           (d)   (d)
#
#  2) use Lanczos to get the (local) ground state, the projector is in shape
#
#        --         --
#        |    |  |   |   
#       L[0]--M--M--R[0]
#        |    |  |   |
#        --         --
#
#  3) put the optimized state in the form as following shape using Svd and normalize
#  psi:
#
#    --A[0]--s[0]--B[0]--
#       |           |
#    
psi = cytnx.UniTensor(cytnx.random.normal([1,d,d,1],1,2),rowrank=2)
shp = psi.shape()
psi_T = psi.get_block_(); psi_T.flatten_() ## flatten to 1d
psi_T, Entemp = eig_Lanczos(psi_T, (L,M,M,R), maxit=maxit);
psi_T.reshape_(*shp)
psi = cytnx.UniTensor(psi_T,rowrank=2)

s0,A,B = cytnx.linalg.Svd_truncate(psi,min(chi,d)) ## Svd
s0/=s0.get_block_().Norm().item() ## normalize

# absorb A[0], B[0] to left & right enviroment.
#
#  L[1]:            R[1]:
#      ----A[0]--       --B[0]----
#      |    |              |     |
#     L[0]--M----       ---M----R[0]
#      |    |              |     |
#      ----A*[0]-       -B*[0]----
#
anet = cytnx.Network("L_AMAH.net")
anet.PutUniTensors(["L","A","A_Conj","M"],[L,A,A.Conj(),M],is_clone=False);
L = anet.Launch(optimal=True)
anet = cytnx.Network("R_AMAH.net")
anet.PutUniTensors(["R","B","B_Conj","M"],[R,B,B.Conj(),M],is_clone=False);
R = anet.Launch(optimal=True)


## Construct n = 1
#  1) again, init a random psi as initial state with shape [d,d,d,d]
#
#      (d)---[psi1]---(d)
#            |    |
#           (d)  (d)
#
#  2) use Lanczos to get the (local) ground state, the projector is in shape
#
#        --         --
#        |    |  |   |   
#       L[1]--M--M--R[1]
#        |    |  |   |
#        --         --
#
#  3) put the optimized state in the form as following shape using Svd and normalize
#  psi:
#
#    --A[1]--s[1]--B[1]--
#       |           |
#
psi = cytnx.UniTensor(cytnx.random.normal([d,d,d,d],0,2),rowrank=2)
shp = psi.shape()
psi_T = psi.get_block_(); psi_T.flatten_() ## flatten to 1d
psi_T, Entemp = eig_Lanczos(psi_T, (L,M,M,R), maxit=maxit);
psi_T.reshape_(*shp)
psi = cytnx.UniTensor(psi_T,rowrank=2)
s1,A,B = cytnx.linalg.Svd_truncate(psi,min(chi,d*d))
s1/=s1.get_block_().Norm().item()

# absorb A[1], B[1] to left & right enviroment.
#
#  L[2]:            R[2]:
#      ----A[1]--       --B[1]----
#      |    |              |     |
#     L[1]--M----       ---M----R[1]
#      |    |              |     |
#      ----A*[1]-       -B*[1]----
#
anet = cytnx.Network("L_AMAH.net")
anet.PutUniTensors(["L","A","A_Conj","M"],[L,A,A.Conj(),M],is_clone=False);
L = anet.Launch(optimal=True)
anet = cytnx.Network("R_AMAH.net")
anet.PutUniTensors(["R","B","B_Conj","M"],[R,B,B.Conj(),M],is_clone=False);
R = anet.Launch(optimal=True)




## Now we are ready for iterative growing system sizes!
for i in range(Niter):

    ## rotate left
    #  1) Absorb s into A and put it in shape
    #            -------------      
    #           /             \     
    #  virt ____|             |____ phys
    #           |             |     
    #           |             |____ virt
    #           \             /     
    #            -------------   
    #  2) Do Svd, and get the growing n+1 right site, and the right singular values sR
    #     [Note] we don't need U!!
    #
    #     --A--s  --B--    =>      --sR--A'  --B--    
    #       |       |                    |     |
    #
    A.set_rowrank(1)
    sR,_,A = cytnx.linalg.Svd(cytnx.Contract(A,s1))



    ## rotate right
    #  1) Absorb s into B and put it in shape
    #            -------------      
    #           /             \     
    #  virt ____|             |____ virt  
    #           |             |     
    #  phys ____|             |        
    #           \             /     
    #            -------------    
    #  2) Do Svd, and get the growing n+1 left site, and the left singular values sL
    #     [Note] we don't need vT!!
    #
    #     --A--  s--B--    =>      --A--   B'--sL    
    #       |       |                |     |
    #
    B.set_rowrank(2)
    sL,B,_ = cytnx.linalg.Svd(cytnx.Contract(s1,B))

    ## now, we change it just to be consistent with the notation in the paper
    #
    #  before:
    #    env-- B'--sL    sR--A' --env
    #          |             |    
    #
    #  after change name:
    #
    #    env-- A--sL     sR--B  --env
    #          |             |
    # 
    A,B = B,A


    ## use sL.s0.sR as new candidate state
    #  [Note] s0 is s_{n-1}, sb = 1/s_{n-1}
    #
    #        --A--[sL--sb--sR]--B--
    #          |                |
    #                 
    #                  to
    #   psi:
    #        --A--[s2]--B--         
    #          |        |       
    #
    sR.set_label(0,1)
    sL.set_label(1,0)
    s0 = 1./s0
    s0.set_labels([0,1])
    s2 = cytnx.Contract(cytnx.Contract(sL,s0),sR)

    s2.set_labels([-10,-11])
    A.set_label(2,-10)
    B.set_label(0,-11)
    psi = cytnx.Contract(cytnx.Contract(A,s2),B)

    ## optimize wave function:
    #  again use Lanczos to get the (local) ground state, the projector is in shape
    #
    #        --         --
    #        |    |  |   |   
    #       L[n]--M--M--R[n]
    #        |    |  |   |
    #        --         --
    #
    shp = psi.shape()
    psi_T = psi.get_block_(); psi_T.flatten_() ## flatten to 1d
    psi_T, Entemp = eig_Lanczos(psi_T, (L,M,M,R), maxit=maxit);
    psi_T.reshape_(*shp)
    psi = cytnx.UniTensor(psi_T,rowrank=2)
    s2,A,B = cytnx.linalg.Svd_truncate(psi,min(chi,psi.shape()[0]*psi.shape()[1]))
    s2/=s2.get_block_().Norm().item()


    ## checking converge:
    #
    #   if (1-<sn|sn-1>) < threadhold, stop.
    #
    if(s2.get_block_().shape()[0] != s1.get_block_().shape()[0]):
        ss = 0
        print("step:%d, increasing bond dim!! dim: %d/%d"%(i,s1.get_block_().shape()[0],chi))
    else:
        ss = abs(cytnx.linalg.Dot(s2.get_block_(),s1.get_block_()).item())
        print("step:%d, diff:%11.11f"%(i,1-ss))
    if(1-ss<1.0e-10): 
        print("[converge!!]")
        break;


    ## absorb into L,R enviroment, and start next iteration. 
    #
    #  L[n+1]:            R[n+1]:
    #      ----A[n]--       --B[n]----
    #      |    |              |     |
    #     L[n]--M----       ---M----R[n]
    #      |    |              |     |
    #      ----A*[n]-       -B*[n]----
    #
    # 
    anet = cytnx.Network("L_AMAH.net")
    anet.PutUniTensors(["L","A","A_Conj","M"],[L,A,A.Conj(),M],is_clone=False);
    L = anet.Launch(optimal=True)

    anet = cytnx.Network("R_AMAH.net")
    anet.PutUniTensors(["R","B","B_Conj","M"],[R,B,B.Conj(),M],is_clone=False);
    R = anet.Launch(optimal=True)

    s0 = s1
    s1 = s2


# use the converged state to get the local energy:
#
#     ----A--s--B----
#     |   |     |   |
#     |   [  H  ]   |   = E_loc
#     |   |     |   |
#     ----A*-s*-B*---
#
anet = cytnx.Network("Measure.net")
anet.PutUniTensors(["psi","psi_conj","M"],[psi,psi,H])
E = anet.Launch(optimal=True).item()
print("ground state E",E)


