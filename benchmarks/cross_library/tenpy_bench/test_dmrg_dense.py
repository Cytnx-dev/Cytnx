"""TeNPy benchmark, algorithm class 1: finite two-site DMRG, dense mode
(no conserved quantum numbers) on the 1D spin-1/2 Heisenberg chain.

CPU only, per the benchmark plan (TeNPy has no GPU backend).

Run timing with `pytest --benchmark-only test_dmrg_dense.py`, memory with
`pytest --memray test_dmrg_dense.py`.
"""
import pytest

from tenpy.algorithms import dmrg
from tenpy.models.spins import SpinChain
from tenpy.networks.mps import MPS

from common.model import BOND_DIM_VALUES, HEISENBERG_J, NUM_SITES_VALUES, N_SWEEPS, GRID_POINT_TIMEOUT_SEC

REFERENCE_ENERGIES = {
    (16, 20): -8.682468456356828,
    (16, 30): -13.111313454922634,
    (16, 50): -21.97181361569925,
    (32, 20): -8.682473319689738,
    (32, 30): -13.111355524192675,
    (32, 50): -21.972106512033726,
    (64, 20): -8.682473334397873,
    (64, 30): -13.11135575848871,
    (64, 50): -21.972110271889434,
}


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


@pytest.mark.timeout(GRID_POINT_TIMEOUT_SEC)
@pytest.mark.parametrize("num_sites", NUM_SITES_VALUES)
@pytest.mark.parametrize("bond_dim", BOND_DIM_VALUES)
def test_dmrg_dense_benchmark(benchmark, bond_dim, num_sites):
    energy = benchmark.pedantic(run_one, args=(bond_dim, num_sites), rounds=1, iterations=1)
    benchmark.extra_info["energy"] = energy
    assert energy == pytest.approx(REFERENCE_ENERGIES[(bond_dim, num_sites)], rel=1e-4)


@pytest.mark.cytnx_memory
@pytest.mark.limit_memory("110 MB")
@pytest.mark.parametrize("num_sites", NUM_SITES_VALUES)
@pytest.mark.parametrize("bond_dim", BOND_DIM_VALUES)
def test_dmrg_dense_memory(bond_dim, num_sites):
    energy = run_one(bond_dim, num_sites)
    assert energy == pytest.approx(REFERENCE_ENERGIES[(bond_dim, num_sites)], rel=1e-4)
