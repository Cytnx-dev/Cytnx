"""TeNPy benchmark, algorithm class 1 (symmetric variant): finite two-site
DMRG with U(1) total-Sz conservation on the 1D spin-1/2 Heisenberg chain.

This exercises TeNPy's `np_conserved` block-sparse tensor backend, which
is the library's flagship optimization for abelian symmetries. CPU only.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tenpy.algorithms import dmrg
from tenpy.models.spins import SpinChain
from tenpy.networks.mps import MPS

from common.model import HEISENBERG_J, N_SWEEPS


def run_one(chi, L):
    model_params = dict(
        L=L, S=0.5, Jx=HEISENBERG_J, Jy=HEISENBERG_J, Jz=HEISENBERG_J,
        bc_MPS="finite", conserve="Sz",
    )
    M = SpinChain(model_params)
    # Total Sz=0 sector (Neel state), required for a conserve='Sz' MPS.
    product_state = (["up", "down"] * (L // 2 + 1))[:L]
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)

    dmrg_params = {
        "mixer": True,
        "trunc_params": {"chi_max": chi, "svd_min": 1e-10},
        "max_sweeps": N_SWEEPS,
        "combine": True,
    }
    eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    E, psi = eng.run()
    return E
