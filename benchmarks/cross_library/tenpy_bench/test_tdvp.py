"""TeNPy benchmark, algorithm class 2: real-time evolution of a 1D
transverse-field Ising chain after a field quench, using TEBD.

(TeNPy also ships a TDVP engine with an almost identical call signature;
TEBD is used here because it is the simpler, more universally-supported
entanglement-growth benchmark and keeps the same comparison point across
all three libraries. See the project report's algorithm-class 2 for the
TDVP variant, which would only change the `tebd.TEBDEngine` line below to
`tdvp.TDVPEngine`.) CPU only.

Run timing with `pytest --benchmark-only test_tdvp.py`, memory with
`pytest --memray test_tdvp.py`.
"""
import pytest

from tenpy.algorithms import tebd
from tenpy.models.tf_ising import TFIChain
from tenpy.networks.mps import MPS

from common.model import CHI_VALUES, L_VALUES, STEP_TIMEOUT_SEC, TFIM_DT, TFIM_HX_FINAL, TFIM_J, TFIM_N_STEPS

REFERENCE_ENERGIES = {
    (16, 20): -9.999703943479117,
    (16, 30): -14.999670104523107,
    (16, 50): -24.99960242661104,
    (32, 20): -9.999703943478044,
    (32, 30): -14.999670104521112,
    (32, 50): -24.999602426607083,
    (64, 20): -9.999703943478044,
    (64, 30): -14.999670104521112,
    (64, 50): -24.999602426607083,
}


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


@pytest.mark.timeout(STEP_TIMEOUT_SEC)
@pytest.mark.parametrize("length", L_VALUES)
@pytest.mark.parametrize("chi", CHI_VALUES)
def test_tdvp_benchmark(benchmark, chi, length):
    energy = benchmark.pedantic(run_one, args=(chi, length), rounds=1, iterations=1)
    benchmark.extra_info["energy"] = float(energy)
    assert float(energy) == pytest.approx(REFERENCE_ENERGIES[(chi, length)], rel=1e-6)


@pytest.mark.limit_memory("40 MB")
def test_tdvp_memory():
    energy = run_one(16, 20)
    assert float(energy) == pytest.approx(REFERENCE_ENERGIES[(16, 20)], rel=1e-6)
