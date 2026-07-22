"""quimb benchmark, algorithm class 2: real-time evolution of a 1D
transverse-field Ising chain after a field quench, using quimb's built-in
`TEBD` engine.

`qtn.ham_1d_ising`/`MPO_ham_ising` build H = j*sum(S^Z_i S^Z_{i+1}) -
bx*sum(S^X_i) using spin-1/2 operators (eigenvalues +-1/2), not Pauli
matrices. To reproduce the same physical Hamiltonian as
`cytnx_bench/test_tebd.py` (H = -hx*sum(PauliX_i) - J*sum(PauliZ_i
PauliZ_{i+1}), Pauli-normalized, eigenvalues +-1), substitute
S^Z = PauliZ/2 and S^X = PauliX/2 and solve for quimb's (j, bx) in terms
of cytnx's (J, hx): j = -4*J, bx = 2*hx.

CPU and GPU code paths are both written; the GPU path moves the initial
MPS to the device array backend before stepping. It cannot be exercised
in this environment (no GPU).

Run timing with `pytest --benchmark-only test_tebd.py`, memory with
`pytest --memray test_tebd.py`.
"""
import pytest

import quimb.tensor as qtn

from common.model import BOND_DIM_VALUES, NUM_SITES_VALUES, GRID_POINT_TIMEOUT_SEC, TFIM_DT, TFIM_HX_FINAL, TFIM_J, TFIM_N_STEPS

DEVICE = "cpu"  # set to "gpu" to exercise the (untested) GPU code path below

# Convert cytnx_bench's Pauli-normalized (J, hx) into quimb's spin-1/2
# convention -- see module docstring.
ISING_J = -4 * TFIM_J
ISING_BX = 2 * TFIM_HX_FINAL

REFERENCE_ENERGIES = {
    (16, 20): -19.000143682959134,
    (16, 30): -29.00015635325205,
    (16, 50): -49.0001810814783,
    (32, 20): -19.000143682959134,
    (32, 30): -29.00015635325205,
    (32, 50): -49.0001810814783,
    (64, 20): -19.000143682959134,
    (64, 30): -29.00015635325205,
    (64, 50): -49.0001810814783,
}


def build(bond_dim, num_sites):
    H = qtn.ham_1d_ising(num_sites, j=ISING_J, bx=ISING_BX, cyclic=False)
    psi0 = qtn.MPS_computational_state("0" * num_sites)
    if DEVICE == "gpu":
        import torch
        psi0.apply_to_arrays(lambda x: torch.as_tensor(x, device="cuda"))
    tebd = qtn.TEBD(psi0, H, dt=TFIM_DT, progbar=False)
    tebd.split_opts["cutoff"] = 1e-10
    tebd.split_opts["max_bond"] = bond_dim
    return tebd


def run_one(bond_dim, num_sites):
    tebd = build(bond_dim, num_sites)
    for _ in range(TFIM_N_STEPS):
        tebd.step(order=2, dt=TFIM_DT)
    H_mpo = qtn.MPO_ham_ising(num_sites, j=ISING_J, bx=ISING_BX, cyclic=False)
    energy = tebd.pt.H @ (H_mpo.apply(tebd.pt))
    return float(energy.real)


@pytest.mark.timeout(GRID_POINT_TIMEOUT_SEC)
@pytest.mark.parametrize("num_sites", NUM_SITES_VALUES)
@pytest.mark.parametrize("bond_dim", BOND_DIM_VALUES)
def test_tebd_benchmark(benchmark, bond_dim, num_sites):
    energy = benchmark.pedantic(run_one, args=(bond_dim, num_sites), rounds=1, iterations=1)
    benchmark.extra_info["energy"] = energy
    assert energy == pytest.approx(REFERENCE_ENERGIES[(bond_dim, num_sites)], rel=1e-6)


@pytest.mark.cytnx_memory
@pytest.mark.limit_memory("100 MB")
@pytest.mark.parametrize("num_sites", NUM_SITES_VALUES)
@pytest.mark.parametrize("bond_dim", BOND_DIM_VALUES)
def test_tebd_memory(bond_dim, num_sites):
    energy = run_one(bond_dim, num_sites)
    assert energy == pytest.approx(REFERENCE_ENERGIES[(bond_dim, num_sites)], rel=1e-6)
