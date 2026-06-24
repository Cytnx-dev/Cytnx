"""TeNPy benchmark, algorithm class 4: variational ground-state search by
gradient descent on the MPS tensors of the 1D Heisenberg chain, with a
hand-derived (manual) gradient instead of automatic differentiation.

TeNPy has no autodiff backend, so the gradient of the Rayleigh quotient
    E(psi) = <psi|H|psi> / <psi|psi>
with respect to a single MPS tensor A_i (holding all other tensors fixed)
is computed analytically rather than via backprop. For an MPS tensor that
sits at the orthogonality center of a mixed-canonical gauge, the standard
result is

    dE/dA_i* = 2 * (H_eff,i(A_i) - E * A_i)

where H_eff,i is the effective one-site Hamiltonian obtained by
contracting the MPO with the left/right boundary environments around site
i -- exactly the operator TeNPy's own DMRG engine builds for its local
Lanczos solve. We reuse TeNPy's `OneSiteH.matvec` to evaluate H_eff,i(A_i)
(this is a *contraction*, not automatic differentiation), then take a
normalized gradient-descent step and renormalize.

This gradient form is specific to TeNPy's `np_conserved` tensor objects
and contraction routines, written independently of the closed-form
gradient used in the quimb (`variational_ad.py`, real autodiff) and Cytnx
(`variational_manual_grad.py`, UniTensor contractions) benchmarks. CPU
only.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tenpy.linalg.np_conserved as npc
from tenpy.algorithms.mps_common import OneSiteH
from tenpy.models.spins import SpinChain
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPOEnvironment

from common.metrics import CSVResultWriter, StepMeasurement, StepTimeoutError, cpu_timed_block, time_limit
from common.model import HEISENBERG_J, N_GRAD_STEPS, STEP_TIMEOUT_SEC, param_grid

LEARNING_RATE = 1e-3


def run_one(chi, L):
    M = SpinChain(dict(
        L=L, S=0.5, Jx=HEISENBERG_J, Jy=HEISENBERG_J, Jz=HEISENBERG_J,
        bc_MPS="finite", conserve=None,
    ))
    sites = M.lat.mps_sites()
    product_state = (["up", "down"] * (L // 2 + 1))[:L]
    psi = MPS.from_random_unitary_evolution(sites, chi, product_state, form="B")
    psi.canonical_form()

    def grad_step():
        env = MPOEnvironment(psi, M.H_MPO, psi)
        energy = None
        for i0 in range(L):
            eff = OneSiteH(env, i0)
            theta = psi.get_theta(i0, n=1)
            h_theta = eff.matvec(theta)
            norm_sq = npc.inner(theta, theta, axes="range", do_conj=True)
            energy = npc.inner(theta, h_theta, axes="range", do_conj=True) / norm_sq
            grad = 2 * (h_theta - energy * theta)
            grad_norm = npc.norm(grad)
            direction = grad / grad_norm if grad_norm > 1e-12 else grad
            new_theta = theta - LEARNING_RATE * direction
            new_theta /= npc.norm(new_theta)
            new_theta.ireplace_label("p0", "p")
            psi.set_B(i0, new_theta, form="Th")
        psi.canonical_form()
        return energy.real

    with cpu_timed_block() as r:
        for _ in range(N_GRAD_STEPS):
            grad_step()
    step_time = r["time_sec"] / N_GRAD_STEPS
    return step_time, r["peak_mem_mb"]


def main(out_csv):
    writer = CSVResultWriter(out_csv)
    for chi, L in param_grid():
        try:
            with time_limit(STEP_TIMEOUT_SEC):
                step_time, peak_mem_mb = run_one(chi, L)
        except StepTimeoutError:
            print(f"[tenpy/variational_manual_grad] chi={chi} L={L} skipped (exceeded {STEP_TIMEOUT_SEC}s)")
            continue
        writer.write(StepMeasurement(
            library="tenpy", algorithm="variational_manual_grad", symmetry="dense",
            device="cpu", backend="manual-grad", L=L, chi=chi,
            step_time_sec=step_time, peak_mem_mb=peak_mem_mb,
        ))
        print(f"[tenpy/variational_manual_grad] chi={chi} L={L} "
              f"time/step={step_time:.4f}s peak_mem={peak_mem_mb:.1f}MB")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results/tenpy_variational.csv"
    main(out)
