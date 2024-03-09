import sys,os
import numpy as np
sys.path.append('example/DMRG/')


from dmrg_two_sites_U1 import *
from numpy import linalg as LA

def test_dmrg_two_sites_XXmodel_U1():

    Nsites = 20 # Number of sites
    chi = 32 # MPS bond dimension
    numsweeps = 6 # number of DMRG sweeps
    maxit = 2 # iterations of Lanczos method

    Es = dmrg_XXmodel_U1(Nsites, chi, numsweeps, maxit)
    H = np.diag(np.ones(Nsites-1),k=1) + np.diag(np.ones(Nsites-1),k=-1)
    D = LA.eigvalsh(H)
    EnExact = 2*sum(D[D < 0])

    assert np.abs(Es[-1]-EnExact)<1e-5
