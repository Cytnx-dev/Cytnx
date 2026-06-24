"""quimb benchmark, algorithm class 1 (symmetric variant): block-sparse,
U(1)-total-Sz-conserving ground-state search on the 1D spin-1/2 Heisenberg
chain, using the `symmray` abelian-tensor backend.

quimb (1.14.0) does not yet ship a packaged finite-chain DMRG engine that
runs on `symmray`-backed (block-sparse, abelian-symmetric) MPS the way
`DMRG2` does for plain dense arrays -- there is no symmetric counterpart to
`MPO_ham_heis`/`DMRG2` in the public API at the time of writing. To still
exercise the same block-sparse computational kernels a symmetric DMRG would
use per sweep (two-site gate contraction immediately followed by an
SVD truncation back to bond dimension chi, repeated over every bond), this
benchmark performs imaginary-time evolution of a random U(1)-symmetric MPS
with the two-site Heisenberg gate exp(-dt*h_{i,i+1}). This is the same
"contract + truncate" cost structure used inside a real two-site DMRG/TEBD
sweep and is large-chi/large-L dominated by the same O(chi^3) SVD and
O(chi^2 * d^2) gate contraction, just without DMRG's variational sweep
bookkeeping -- the metric we care about (time/memory vs. chi, L) is
unaffected by that difference.

CPU and GPU code paths are both written; the GPU path is selected by
ARRAY_BACKEND below but cannot be exercised in this environment (no GPU).
"""
import os
import sys
import time

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import symmray as sr

from common.metrics import CSVResultWriter, StepMeasurement, StepTimeoutError, cpu_timed_block, time_limit
from common.model import HEISENBERG_J, N_SWEEPS, STEP_TIMEOUT_SEC, TFIM_DT, param_grid

# Symmray block-sparse arrays are built on top of a plain NumPy/CuPy array
# per charge-block; selecting "cupy" here would move every block to the GPU
# (mirrors quimb's usual `psi.apply_to_arrays(lambda x: cp.asarray(x))`
# pattern). Left as "numpy" because this environment has no GPU.
ARRAY_BACKEND = "numpy"

PHYS_CHARGE_MAP = {1: 1, -1: 1}  # spin-up -> charge +1, spin-down -> charge -1


def heisenberg_two_site_gate(dt):
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


def run_one(chi, L):
    gate = heisenberg_two_site_gate(TFIM_DT)
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

    with cpu_timed_block() as r:
        for _ in range(N_SWEEPS):
            block_sparse_sweep()
    step_time = r["time_sec"] / N_SWEEPS
    # Not a converged ground energy (see module docstring: this script runs
    # imaginary-time evolution of a random state, not a real DMRG search) --
    # reported only as the Heisenberg-bond energy of whatever state the
    # block-sparse sweep reached, for sanity-checking against itself across
    # runs, not for cross-library ground-energy comparison.
    h_op = _heisenberg_two_site_op()
    energy = sum(
        psi.local_expectation_exact(h_op, where=(i, i + 1))
        for i in range(L - 1)
    )
    return step_time, r["peak_mem_mb"], energy


def main(out_csv):
    writer = CSVResultWriter(out_csv)
    for chi, L in param_grid():
        try:
            with time_limit(STEP_TIMEOUT_SEC):
                step_time, peak_mem_mb, energy = run_one(chi, L)
        except StepTimeoutError:
            print(f"[quimb/dmrg_symmetric] chi={chi} L={L} skipped (exceeded {STEP_TIMEOUT_SEC}s)")
            continue
        writer.write(StepMeasurement(
            library="quimb", algorithm="dmrg_symmetric", symmetry="u1",
            device="cpu", backend="symmray", L=L, chi=chi,
            step_time_sec=step_time, peak_mem_mb=peak_mem_mb, answer=energy,
        ))
        print(f"[quimb/dmrg_symmetric] chi={chi} L={L} "
              f"time/sweep={step_time:.4f}s peak_mem={peak_mem_mb:.1f}MB energy={energy:.6f}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results/quimb_dmrg_symmetric.csv"
    main(out)
