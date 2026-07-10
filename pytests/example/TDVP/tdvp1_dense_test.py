import sys,os
import numpy as np
sys.path.append('example/TDVP/')

from tdvp1_dense import *
from numpy import linalg as LA


def run_product_state_tdvp(chi, time_step):
    # Prepare a deterministic product state with overlap on both local Sz eigenstates.
    Nsites = 8 # Number of sites
    d = 2
    MPS_x = prepare_x_init_MPS(Nsites, chi, d)

    # simulate ground state by imaginary time evolution by tdvp
    J = 0.0
    Jz = 0.0
    hx = 0.0
    hz = -1.0
    tau = -1.0j
    # prepare up state
    return tdvp1_XXZmodel_dense(J, Jz, hx, hz, MPS_x, chi, tau, time_step)


def test_tdvp1_dense():
    Nsites = 8 # Number of sites
    chi = 1 # MPS bond dimension
    time_step = 17
    As, Es = run_product_state_tdvp(chi, time_step)
    error = np.abs(Es[-1]-(-1.0*Nsites))

    # For this test Hamiltonian, H = -sum_i Sz_i.  The |x> product state has
    # ground-state weight 1 / 2**Nsites.  The first excited sector is Nsites-fold
    # degenerate with gap 2, so after imaginary time beta the leading energy
    # error is O(2 * Nsites * exp(-2 * beta)).
    #
    # The exact imaginary-time trajectory stays in the chi = 1 product-state
    # manifold.  For Nsites = 8 and time_step = 17 this estimate is 2.742e-14.
    # The test asserts that the TDVP error is within a factor of 20 of this
    # scale.  As of commit 640bb4bfcc989bd9c1126ee13b3fd1c3920803e9, the
    # observed error is at roundoff.
    leading_energy_error = 2.0 * Nsites * np.exp(-2.0 * time_step)
    assert error < 20.0 * leading_energy_error


def test_tdvp1_dense_truncates_null_padded_product_state():
    chi1_As, chi1_Es = run_product_state_tdvp(chi=1, time_step=17)
    chi2_As, chi2_Es = run_product_state_tdvp(chi=2, time_step=17)

    assert [tensor.shape() for tensor in chi2_As[-1]] == [tensor.shape() for tensor in chi1_As[-1]]
    assert np.abs(chi2_Es[-1] - chi1_Es[-1]) < 1e-12
