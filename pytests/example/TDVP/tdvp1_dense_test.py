import sys,os
import numpy as np
sys.path.append('example/TDVP/')

from tdvp1_dense import *
from numpy import linalg as LA

def test_tdvp1_dense():
    #prepare random MPS
    Nsites = 20 # Number of sites
    chi = 16 # MPS bond dimension
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
    error = np.abs(Es[-1]-(-1.0*Nsites))
    assert error < 1e-6
