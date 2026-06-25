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

from common.model import CHI_VALUES, HEISENBERG_J, L_VALUES, N_SWEEPS, STEP_TIMEOUT_SEC

REFERENCE_ENERGIES = {
    (16, 20): -8.682468456352291,
    (16, 30): -13.111313454915052,
    (16, 50): -21.97181361378568,
    (32, 20): -8.682473319689738,
    (32, 30): -13.111355524192675,
    (32, 50): -21.972106507515218,
    (64, 20): -8.682473334397873,
    (64, 30): -13.11135575848871,
    (64, 50): -21.972110271671795,
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


@pytest.mark.timeout(STEP_TIMEOUT_SEC)
@pytest.mark.parametrize("length", L_VALUES)
@pytest.mark.parametrize("chi", CHI_VALUES)
def test_dmrg_dense_benchmark(benchmark, chi, length):
    energy = benchmark.pedantic(run_one, args=(chi, length), rounds=1, iterations=1)
    benchmark.extra_info["energy"] = energy
    assert energy == pytest.approx(REFERENCE_ENERGIES[(chi, length)], rel=1e-4)


@pytest.mark.limit_memory("50 MB")
def test_dmrg_dense_memory():
    energy = run_one(16, 20)
    assert energy == pytest.approx(REFERENCE_ENERGIES[(16, 20)], rel=1e-4)
