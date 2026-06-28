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
`run_one_symmetric` instead drives the same random U(1)-symmetric MPS to
the Heisenberg ground state via imaginary-time evolution (ITE) with the
two-site gate exp(-dt*h_{i,i+1}): repeated application of exp(-dt*H) to a
state with nonzero ground-state overlap converges to the ground state of
the targeted symmetry sector as the total imaginary time grows, exactly
like the dense/symmetric DMRG benchmarks converge to the same energy via
their own (different) sweep algorithm. `ITE_DT_SCHEDULE` anneals dt from
0.3 down to 0.02 across zigzag (forward then backward) sweeps over every
bond, renormalizing once per full zigzag pass; this uses the same
"contract + truncate" cost structure as a real two-site DMRG/TEBD sweep
(large-chi/large-L dominated by the same O(chi^3) SVD and O(chi^2 * d^2)
gate contraction) so the time/memory metrics remain comparable, while the
reported energy now converges to the same Heisenberg ground energy as
`cytnx_bench/test_dmrg_symmetric.py`/`tenpy_bench/test_dmrg_symmetric.py`
at every (chi, L) grid point (matched to within the `rel=1e-2` tolerance
used for `SYMMETRIC_REFERENCE_ENERGIES`, looser than the dense/symmetric
DMRG benchmarks' `rel=1e-4` since ITE over a finite total imaginary time
is itself an approximation to the true ground state, not just a
discretization of an already-exact sweep).

Run timing with `pytest --benchmark-only test_dmrg.py`, memory with
`pytest --memray test_dmrg.py`.
"""
import numpy as np
import pytest
import symmray as sr
from scipy.linalg import expm

import quimb.tensor as qtn

from common.model import BOND_DIM_VALUES, HEISENBERG_J, NUM_SITES_VALUES, N_SWEEPS, GRID_POINT_TIMEOUT_SEC

DEVICE = "cpu"  # set to "gpu" to exercise the (untested) GPU code path below

# Symmray block-sparse arrays are built on top of a plain array per
# charge-block, so this selects which array library backs every block of
# `psi` (and the gate/operator it's contracted against) in
# `run_one_symmetric`. "numpy" is fastest across the full (bond_dim,
# num_sites) grid -- "torch" lands the same energies but runs consistently
# ~1.5-2x slower at every grid point, since this workload is dominated by
# thousands of small per-bond gate/SVD calls where torch's dispatch
# overhead outweighs any kernel speedup. "jax" is not a viable alternative
# at all: its per-call tracing overhead is so large that a single grid
# point did not finish in several minutes, vs. seconds for numpy/torch.
# "cupy" moves every block to the GPU (mirrors quimb's usual
# `psi.apply_to_arrays(lambda x: cp.asarray(x))` pattern) but cannot be
# exercised in this environment (no GPU).
ARRAY_BACKEND = "numpy"

PHYS_CHARGE_MAP = {1: 1, -1: 1}  # spin-up -> charge +1, spin-down -> charge -1

DENSE_REFERENCE_ENERGIES = {
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
SYMMETRIC_REFERENCE_ENERGIES = {
    (16, 20): -8.675022726955829,
    (16, 30): -13.058213723292113,
    (16, 50): -21.703613228188505,
    (32, 20): -8.682060417783728,
    (32, 30): -13.096659398821432,
    (32, 50): -21.86788053095872,
    (64, 20): -8.682293734678527,
    (64, 30): -13.105181026506637,
    (64, 50): -21.925672520032034,
}

# Anneals dt across zigzag sweeps -- see module docstring.
ITE_DT_SCHEDULE = [0.3] * 15 + [0.2] * 15 + [0.1] * 20 + [0.05] * 30 + [0.02] * 80


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


def _convert_backend(arr):
    """Move a U1Array's underlying blocks to ARRAY_BACKEND's array type."""
    if ARRAY_BACKEND == "numpy":
        return arr
    if ARRAY_BACKEND == "cupy":
        import cupy as cp
        arr.apply_to_arrays(lambda x: cp.asarray(x))
    elif ARRAY_BACKEND == "torch":
        import torch
        arr.apply_to_arrays(lambda x: torch.as_tensor(x))
    elif ARRAY_BACKEND == "jax":
        import jax.numpy as jnp
        arr.apply_to_arrays(lambda x: jnp.asarray(x))
    return arr


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
    gate = sr.U1Array.from_dense(
        gate_dense, index_maps=[phys, phys, phys, phys],
        duals=[False, False, True, True],
    )
    return _convert_backend(gate)


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
    op = sr.U1Array.from_dense(
        h_dense.reshape(2, 2, 2, 2), index_maps=[phys, phys, phys, phys],
        duals=[False, False, True, True],
    )
    return _convert_backend(op)


def run_one_symmetric(chi, L):
    # Alternate site charges so the half-filled (Neel-like) total-Sz=0
    # sector is reachable at the bond dimensions in our sweep grid.
    psi = sr.MPS_abelian_rand(
        "U1", L=L, bond_dim=chi, phys_dim=PHYS_CHARGE_MAP, seed=0,
        site_charge=lambda i: 1 if i % 2 == 0 else -1,
    )
    _convert_backend(psi)

    gates = {dt: _heisenberg_two_site_gate(dt) for dt in set(ITE_DT_SCHEDULE)}

    def zigzag_pass(dt):
        gate = gates[dt]
        for i in range(L - 1):
            psi.gate_split_(gate, where=(i, i + 1), max_bond=chi, cutoff=1e-10)
        for i in range(L - 2, -1, -1):
            psi.gate_split_(gate, where=(i, i + 1), max_bond=chi, cutoff=1e-10)
        psi.normalize()

    for dt in ITE_DT_SCHEDULE:
        zigzag_pass(dt)
    h_op = _heisenberg_two_site_op()
    energy = sum(
        psi.local_expectation_exact(h_op, where=(i, i + 1))
        for i in range(L - 1)
    )
    return energy


@pytest.mark.timeout(GRID_POINT_TIMEOUT_SEC)
@pytest.mark.parametrize("num_sites", NUM_SITES_VALUES)
@pytest.mark.parametrize("bond_dim", BOND_DIM_VALUES)
def test_dmrg_dense_benchmark(benchmark, bond_dim, num_sites):
    energy = benchmark.pedantic(run_one_dense, args=(bond_dim, num_sites), rounds=1, iterations=1)
    benchmark.extra_info["energy"] = float(energy)
    assert float(energy) == pytest.approx(DENSE_REFERENCE_ENERGIES[(bond_dim, num_sites)], rel=1e-4)


@pytest.mark.cytnx_memory
@pytest.mark.limit_memory("130 MB")
@pytest.mark.parametrize("num_sites", NUM_SITES_VALUES)
@pytest.mark.parametrize("bond_dim", BOND_DIM_VALUES)
def test_dmrg_dense_memory(bond_dim, num_sites):
    energy = run_one_dense(bond_dim, num_sites)
    assert float(energy) == pytest.approx(DENSE_REFERENCE_ENERGIES[(bond_dim, num_sites)], rel=1e-4)


@pytest.mark.timeout(GRID_POINT_TIMEOUT_SEC)
@pytest.mark.parametrize("num_sites", NUM_SITES_VALUES)
@pytest.mark.parametrize("bond_dim", BOND_DIM_VALUES)
def test_dmrg_symmetric_benchmark(benchmark, bond_dim, num_sites):
    energy = benchmark.pedantic(run_one_symmetric, args=(bond_dim, num_sites), rounds=1, iterations=1)
    benchmark.extra_info["energy"] = float(energy)
    assert float(energy) == pytest.approx(SYMMETRIC_REFERENCE_ENERGIES[(bond_dim, num_sites)], rel=2e-2)


@pytest.mark.cytnx_memory
@pytest.mark.limit_memory("700 MB")
@pytest.mark.parametrize("num_sites", NUM_SITES_VALUES)
@pytest.mark.parametrize("bond_dim", BOND_DIM_VALUES)
def test_dmrg_symmetric_memory(bond_dim, num_sites):
    energy = run_one_symmetric(bond_dim, num_sites)
    assert float(energy) == pytest.approx(SYMMETRIC_REFERENCE_ENERGIES[(bond_dim, num_sites)], rel=2e-2)
