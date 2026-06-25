"""TeNPy benchmark, algorithm class 2: real-time evolution of a 1D
transverse-field Ising chain after a field quench, using TEBD.

(TeNPy also ships a TDVP engine with an almost identical call signature;
TEBD is used here because it is the simpler, more universally-supported
entanglement-growth benchmark and keeps the same comparison point across
all three libraries. See the project report's algorithm-class 2 for the
TDVP variant, which would only change the `tebd.TEBDEngine` line below to
`tdvp.TDVPEngine`.) CPU only.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tenpy.algorithms import tebd
from tenpy.models.tf_ising import TFIChain
from tenpy.networks.mps import MPS

from common.model import TFIM_DT, TFIM_HX_FINAL, TFIM_J, TFIM_N_STEPS


def run_one(chi, L):
    model_params = dict(L=L, J=TFIM_J, g=TFIM_HX_FINAL, bc_MPS="finite", conserve=None)
    M = TFIChain(model_params)
    # Start fully polarized along x (paramagnetic ground state of the
    # pre-quench Hamiltonian at large field) then quench to g=TFIM_HX_FINAL.
    psi = MPS.from_product_state(M.lat.mps_sites(), ["up"] * L, bc=M.lat.bc_MPS)

    tebd_params = {
        "N_steps": 1,
        "dt": TFIM_DT,
        "order": 2,
        "trunc_params": {"chi_max": chi, "svd_min": 1e-10},
    }
    eng = tebd.TEBDEngine(psi, M, tebd_params)

    for _ in range(TFIM_N_STEPS):
        eng.run()
    energy = M.H_MPO.expectation_value(psi)
    return energy
