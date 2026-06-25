"""TeNPy benchmark, algorithm class 1: finite two-site DMRG, dense mode
(no conserved quantum numbers) on the 1D spin-1/2 Heisenberg chain.

CPU only, per the benchmark plan (TeNPy has no GPU backend).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tenpy.algorithms import dmrg
from tenpy.models.spins import SpinChain
from tenpy.networks.mps import MPS

from common.model import HEISENBERG_J, N_SWEEPS


def run_one(chi, L, dmrg_chi_max=None):
    model_params = dict(
        L=L, S=0.5, Jx=HEISENBERG_J, Jy=HEISENBERG_J, Jz=HEISENBERG_J,
        bc_MPS="finite", conserve=None,
    )
    M = SpinChain(model_params)
    product_state = (["up", "down"] * (L // 2 + 1))[:L]
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)

    dmrg_params = {
        "mixer": True,
        "trunc_params": {"chi_max": dmrg_chi_max or chi, "svd_min": 1e-10},
        "max_sweeps": N_SWEEPS,
        "combine": True,
    }
    eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    E, psi = eng.run()
    return E
