"""TeNPy benchmark, algorithm class 1 (symmetric variant): finite two-site
DMRG with U(1) total-Sz conservation on the 1D spin-1/2 Heisenberg chain.

This exercises TeNPy's `np_conserved` block-sparse tensor backend, which
is the library's flagship optimization for abelian symmetries. CPU only.

Run timing with `pytest --benchmark-only test_dmrg_symmetric.py`, memory
with `pytest --memray test_dmrg_symmetric.py`.
"""
import pytest

from tenpy.algorithms import dmrg
from tenpy.models.spins import SpinChain
from tenpy.networks.mps import MPS

from common.model import CHI_VALUES, HEISENBERG_J, L_VALUES, N_SWEEPS, STEP_TIMEOUT_SEC

REFERENCE_ENERGIES = {
    (16, 20): -8.682468456352254,
    (16, 30): -13.111313454915095,
    (16, 50): -21.971813613863763,
    (32, 20): -8.682473319689738,
    (32, 30): -13.111355524202278,
    (32, 50): -21.972106507466883,
    (64, 20): -8.682473334397892,
    (64, 30): -13.11135575848872,
    (64, 50): -21.972110271683714,
}


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


@pytest.mark.timeout(STEP_TIMEOUT_SEC)
@pytest.mark.parametrize("length", L_VALUES)
@pytest.mark.parametrize("chi", CHI_VALUES)
def test_dmrg_symmetric_benchmark(benchmark, chi, length):
    energy = benchmark.pedantic(run_one, args=(chi, length), rounds=1, iterations=1)
    benchmark.extra_info["energy"] = energy
    assert energy == pytest.approx(REFERENCE_ENERGIES[(chi, length)], rel=1e-4)


@pytest.mark.cytnx_memory
@pytest.mark.limit_memory("40 MB")
def test_dmrg_symmetric_memory():
    energy = run_one(16, 20)
    assert energy == pytest.approx(REFERENCE_ENERGIES[(16, 20)], rel=1e-4)
