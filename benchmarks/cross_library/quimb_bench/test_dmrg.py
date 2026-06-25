"""quimb benchmark, algorithm class 1: finite two-site DMRG, dense mode
(no conserved quantum numbers) on the 1D spin-1/2 Heisenberg chain, using
quimb's built-in `DMRG2` engine, and a symmetric variant exercising the
`symmray` block-sparse backend.

CPU and GPU code paths are both written; the GPU path moves every MPO/MPS
tensor to the device array backend before the sweep (mirrors quimb's
documented `to_backend`/`apply_to_arrays` pattern with cupy or torch). It
cannot be exercised in this environment (no GPU).

quimb (1.14.0) does not yet ship a packaged finite-chain DMRG engine that
runs on `symmray`-backed (block-sparse, abelian-symmetric) MPS the way
`DMRG2` does for plain dense arrays -- there is no symmetric counterpart to
`MPO_ham_heis`/`DMRG2` in the public API at the time of writing. To still
exercise the same block-sparse computational kernels a symmetric DMRG
would use per sweep (two-site gate contraction immediately followed by an
SVD truncation back to bond dimension chi, repeated over every bond),
`run_one_symmetric` performs imaginary-time evolution of a random
U(1)-symmetric MPS with the two-site Heisenberg gate exp(-dt*h_{i,i+1}).
This is the same "contract + truncate" cost structure used inside a real
two-site DMRG/TEBD sweep and is large-chi/large-L dominated by the same
O(chi^3) SVD and O(chi^2 * d^2) gate contraction, just without DMRG's
variational sweep bookkeeping -- the metric we care about (time/memory vs.
chi, L) is unaffected by that difference. The reported energy is not a
converged ground energy, only the Heisenberg-bond energy of whatever state
the block-sparse sweep reached, so its reference values are a
reproducibility check (seeded MPS), not a ground-energy correctness check.

Run timing with `pytest --benchmark-only test_dmrg.py`, memory with
`pytest --memray test_dmrg.py`.
"""
import numpy as np
import pytest
import symmray as sr
from scipy.linalg import expm

import quimb.tensor as qtn

from common.model import CHI_VALUES, HEISENBERG_J, L_VALUES, N_SWEEPS, STEP_TIMEOUT_SEC, TFIM_DT

DEVICE = "cpu"  # set to "gpu" to exercise the (untested) GPU code path below

# Symmray block-sparse arrays are built on top of a plain NumPy/CuPy array
# per charge-block; selecting "cupy" here would move every block to the GPU
# (mirrors quimb's usual `psi.apply_to_arrays(lambda x: cp.asarray(x))`
# pattern). Left as "numpy" because this environment has no GPU.
ARRAY_BACKEND = "numpy"

PHYS_CHARGE_MAP = {1: 1, -1: 1}  # spin-up -> charge +1, spin-down -> charge -1

DENSE_REFERENCE_ENERGIES = {
    (16, 20): -8.682468366590122,
    (16, 30): -13.111312475497847,
    (16, 50): -21.971804927253896,
    (32, 20): -8.682473318039497,
    (32, 30): -13.111355518809908,
    (32, 50): -21.972106192736593,
    (64, 20): -8.682473330915784,
    (64, 30): -13.111355751853354,
    (64, 50): -21.972110252864823,
}
SYMMETRIC_REFERENCE_ENERGIES = {
    (16, 20): -0.564136128480123,
    (16, 30): -1.3649283608399005,
    (16, 50): -1.8056473018804011,
    (32, 20): -0.04873844749336939,
    (32, 30): -1.049817168087423,
    (32, 50): -0.8520392861421351,
    (64, 20): -0.23048590017638557,
    (64, 30): -2.0860038446373554,
    (64, 50): -0.747960030333261,
}


def build_dense(chi, L):
    H = qtn.MPO_ham_heis(L, j=HEISENBERG_J, cyclic=False)
    if DEVICE == "gpu":
        import torch
        H.apply_to_arrays(lambda x: torch.as_tensor(x, device="cuda"))
    dmrg = qtn.DMRG2(H, bond_dims=[chi], cutoffs=1e-10)
    return dmrg


def run_one_dense(chi, L):
    dmrg = build_dense(chi, L)
    dmrg.solve(tol=1e-6, max_sweeps=N_SWEEPS, verbosity=0)
    return dmrg.energy


def _heisenberg_two_site_gate(dt):
    """Charge-conserving exp(-dt * J * S_i.S_j) two-site gate as a U1Array.

    In the {up(+1), down(-1)} Sz basis, J*S_i.S_j is exactly block-diagonal
    in total Sz: the (up,up) and (down,down) sectors (total charge +-2) are
    1x1 blocks, and the (up,down)/(down,up) sector (total charge 0) is a 2x2
    block. `expm` of this dense 4x4 matrix preserves that same block
    structure, so converting it through `U1Array.from_dense` recovers a
    genuinely sparse (6 nonzero blocks out of a dense 16-element 2x2x2x2
    tensor) charge-conserving gate.
    """
    phys = [1, -1]
    h_dense = (HEISENBERG_J / 4.0) * np.array([
        [1, 0, 0, 0],
        [0, -1, 2, 0],
        [0, 2, -1, 0],
        [0, 0, 0, 1],
    ], dtype=float)
    gate_dense = expm(-dt * h_dense).reshape(2, 2, 2, 2)
    return sr.U1Array.from_dense(
        gate_dense, index_maps=[phys, phys, phys, phys],
        duals=[False, False, True, True],
    )


def _heisenberg_two_site_op():
    """Plain (non-exponentiated) two-site Heisenberg coupling, for computing
    <psi|H|psi> via `local_expectation` -- not used to evolve `psi`."""
    phys = [1, -1]
    h_dense = (HEISENBERG_J / 4.0) * np.array([
        [1, 0, 0, 0],
        [0, -1, 2, 0],
        [0, 2, -1, 0],
        [0, 0, 0, 1],
    ], dtype=float)
    return sr.U1Array.from_dense(
        h_dense.reshape(2, 2, 2, 2), index_maps=[phys, phys, phys, phys],
        duals=[False, False, True, True],
    )


def run_one_symmetric(chi, L):
    gate = _heisenberg_two_site_gate(TFIM_DT)
    # Alternate site charges so the half-filled (Neel-like) total-Sz=0
    # sector is reachable at the bond dimensions in our sweep grid.
    psi = sr.MPS_abelian_rand(
        "U1", L=L, bond_dim=chi, phys_dim=PHYS_CHARGE_MAP, seed=0,
        site_charge=lambda i: 1 if i % 2 == 0 else -1,
    )
    if ARRAY_BACKEND != "numpy":
        import cupy as cp
        psi.apply_to_arrays(lambda x: cp.asarray(x))

    def block_sparse_sweep():
        for i in range(L - 1):
            psi.gate_split_(gate, where=(i, i + 1), max_bond=chi, cutoff=1e-10)

    for _ in range(N_SWEEPS):
        block_sparse_sweep()
    h_op = _heisenberg_two_site_op()
    energy = sum(
        psi.local_expectation_exact(h_op, where=(i, i + 1))
        for i in range(L - 1)
    )
    return energy


@pytest.mark.timeout(STEP_TIMEOUT_SEC)
@pytest.mark.parametrize("length", L_VALUES)
@pytest.mark.parametrize("chi", CHI_VALUES)
def test_dmrg_dense_benchmark(benchmark, chi, length):
    energy = benchmark.pedantic(run_one_dense, args=(chi, length), rounds=1, iterations=1)
    benchmark.extra_info["energy"] = float(energy)
    assert float(energy) == pytest.approx(DENSE_REFERENCE_ENERGIES[(chi, length)], rel=1e-4)


@pytest.mark.limit_memory("80 MB")
def test_dmrg_dense_memory():
    energy = run_one_dense(16, 20)
    assert float(energy) == pytest.approx(DENSE_REFERENCE_ENERGIES[(16, 20)], rel=1e-4)


@pytest.mark.timeout(STEP_TIMEOUT_SEC)
@pytest.mark.parametrize("length", L_VALUES)
@pytest.mark.parametrize("chi", CHI_VALUES)
def test_dmrg_symmetric_benchmark(benchmark, chi, length):
    energy = benchmark.pedantic(run_one_symmetric, args=(chi, length), rounds=1, iterations=1)
    benchmark.extra_info["energy"] = float(energy)
    assert float(energy) == pytest.approx(SYMMETRIC_REFERENCE_ENERGIES[(chi, length)], rel=1e-6)


@pytest.mark.limit_memory("700 MB")
def test_dmrg_symmetric_memory():
    energy = run_one_symmetric(16, 20)
    assert float(energy) == pytest.approx(SYMMETRIC_REFERENCE_ENERGIES[(16, 20)], rel=1e-6)
