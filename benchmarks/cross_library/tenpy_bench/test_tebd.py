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
`cytnx_bench/test_tebd.py`'s and `quimb_bench/test_tebd.py`'s H =
-hx*sum(PauliX_i) - J*sum(PauliZ_i PauliZ_{i+1}) (coupling on Z, field on
X). `_TFIChainZCoupling` below overrides `init_terms` to swap the two
Pauli operators back, so this script's Hamiltonian uses the same axis
convention as the other two libraries and all three can start from the
literal same Sigmaz computational-basis ("up") product state.

Run timing with `pytest --benchmark-only test_tebd.py`, memory with
`pytest --memray test_tebd.py`.
"""
import numpy as np
import pytest

from tenpy.algorithms import tebd
from tenpy.models.tf_ising import TFIChain
from tenpy.networks.mps import MPS

from common.model import BOND_DIM_VALUES, NUM_SITES_VALUES, GRID_POINT_TIMEOUT_SEC, TFIM_DT, TFIM_HX_FINAL, TFIM_J, TFIM_N_STEPS


class _TFIChainZCoupling(TFIChain):
    """TFIChain with the coupling on Sigmaz and the field on Sigmax, matching
    cytnx_bench/test_tebd.py's and quimb_bench/test_tebd.py's H =
    -hx*sum(X_i) - J*sum(Z_i Z_{i+1}) convention instead of TFIChain's own
    H = -J*sum(X_i X_{i+1}) - g*sum(Z_i)."""

    def init_terms(self, model_params):
        J = np.asarray(model_params.get('J', 1.0, 'real_or_array'))
        g = np.asarray(model_params.get('g', 1.0, 'real_or_array'))
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-g, u, 'Sigmax')
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-J, u1, 'Sigmaz', u2, 'Sigmaz', dx)


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


def run_one(bond_dim, num_sites):
    model_params = dict(L=num_sites, J=TFIM_J, g=TFIM_HX_FINAL, bc_MPS="finite", conserve=None)
    M = _TFIChainZCoupling(model_params)
    # Sigmaz "up" eigenstate -- the same computational-basis product state
    # used by cytnx_bench/test_tebd.py and quimb_bench/test_tebd.py.
    up = np.array([1.0, 0.0])
    psi = MPS.from_product_state(M.lat.mps_sites(), [up] * num_sites, bc=M.lat.bc_MPS)

    tebd_params = {
        "N_steps": 1,
        "dt": TFIM_DT,
        "order": 2,
        "trunc_params": {"chi_max": bond_dim, "svd_min": 1e-10},
    }
    eng = tebd.TEBDEngine(psi, M, tebd_params)

    for _ in range(TFIM_N_STEPS):
        eng.run()
    energy = M.H_MPO.expectation_value(psi)
    return energy


@pytest.mark.timeout(GRID_POINT_TIMEOUT_SEC)
@pytest.mark.parametrize("num_sites", NUM_SITES_VALUES)
@pytest.mark.parametrize("bond_dim", BOND_DIM_VALUES)
def test_tebd_benchmark(benchmark, bond_dim, num_sites):
    energy = benchmark.pedantic(run_one, args=(bond_dim, num_sites), rounds=1, iterations=1)
    benchmark.extra_info["energy"] = float(energy)
    assert float(energy) == pytest.approx(REFERENCE_ENERGIES[(bond_dim, num_sites)], rel=1e-6)


@pytest.mark.cytnx_memory
@pytest.mark.limit_memory("90 MB")
@pytest.mark.parametrize("num_sites", NUM_SITES_VALUES)
@pytest.mark.parametrize("bond_dim", BOND_DIM_VALUES)
def test_tebd_memory(bond_dim, num_sites):
    energy = run_one(bond_dim, num_sites)
    assert float(energy) == pytest.approx(REFERENCE_ENERGIES[(bond_dim, num_sites)], rel=1e-6)
