"""TeNPy benchmark, algorithm class 2: real-time evolution of a 1D
transverse-field Ising chain after a field quench, using TEBD.

(TeNPy also ships a TDVP engine with an almost identical call signature;
TEBD is used here because it is the simpler, more universally-supported
entanglement-growth benchmark and keeps the same comparison point across
all three libraries. See the project report's algorithm-class 2 for the
TDVP variant, which would only change the `tebd.TEBDEngine` line below to
`tdvp.TDVPEngine`.) CPU only.

`TFIChain`'s `init_terms` builds H = -J*sum(Sigmax_i Sigmax_{i+1}) -
g*sum(Sigmaz_i), i.e. the coupling and field axes are swapped relative to
`cytnx_bench/test_tebd.py`'s H = -hx*sum(PauliX_i) - J*sum(PauliZ_i
PauliZ_{i+1}) (coupling on Z, field on X). To prepare the same physical
initial state as cytnx's (a product state aligned along the coupling
axis), the per-site state here must be a Sigmax eigenstate, not a
Sigmaz eigenstate -- `["up"]*L` (a Sigmaz eigenstate, TeNPy's field axis)
would instead start aligned with the field, a different physical setup.

Run timing with `pytest --benchmark-only test_tdvp.py`, memory with
`pytest --memray test_tdvp.py`.
"""
import numpy as np
import pytest

from tenpy.algorithms import tebd
from tenpy.models.tf_ising import TFIChain
from tenpy.networks.mps import MPS

from common.model import CHI_VALUES, L_VALUES, STEP_TIMEOUT_SEC, TFIM_DT, TFIM_HX_FINAL, TFIM_J, TFIM_N_STEPS

REFERENCE_ENERGIES = {
    (16, 20): -19.00014659257969,
    (16, 30): -29.000162658620344,
    (16, 50): -49.00019479070228,
    (32, 20): -19.00014659257974,
    (32, 30): -29.000162658620344,
    (32, 50): -49.000194790702395,
    (64, 20): -19.00014659257974,
    (64, 30): -29.000162658620344,
    (64, 50): -49.000194790702395,
}


def run_one(chi, L):
    model_params = dict(L=L, J=TFIM_J, g=TFIM_HX_FINAL, bc_MPS="finite", conserve=None)
    M = TFIChain(model_params)
    # Sigmax eigenstate -- aligned with TeNPy's coupling axis, matching
    # cytnx's computational-basis state along its own coupling axis.
    plus_x = np.array([1.0, 1.0]) / np.sqrt(2)
    psi = MPS.from_product_state(M.lat.mps_sites(), [plus_x] * L, bc=M.lat.bc_MPS)

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


@pytest.mark.cytnx_memory
@pytest.mark.limit_memory("40 MB")
def test_tdvp_memory():
    energy = run_one(16, 20)
    assert float(energy) == pytest.approx(REFERENCE_ENERGIES[(16, 20)], rel=1e-6)
