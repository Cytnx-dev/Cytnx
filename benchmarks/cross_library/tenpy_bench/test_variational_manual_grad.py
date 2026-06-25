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
(this is a *contraction*, not automatic differentiation), then take an
unnormalized gradient-descent step and renormalize.

Each updated site is immediately QR-left-canonicalized and the leftover
upper-triangular factor is folded into the next (not-yet-visited) site's
tensor before that site's theta is read out, mirroring the per-site
re-gauging that `test_dmrg_dense.py`'s effective Hamiltonian already
assumes. `MPOEnvironment`'s LP/RP caches are populated lazily, so they
pick up the newly re-gauged tensors as the sweep reaches each site
without having to be rebuilt by hand. A single trailing
`psi.canonical_form()` call restores TeNPy's right-canonical 'B' form
everywhere for the next sweep's lazy environment caching to remain valid.

This gradient form is specific to TeNPy's `np_conserved` tensor objects
and contraction routines, written independently of the closed-form
gradient used in the quimb (`test_variational_ad.py`, real autodiff) and
Cytnx (`test_variational_manual_grad.py`, UniTensor contractions)
benchmarks. CPU only.

Run timing with `pytest --benchmark-only test_variational_manual_grad.py`,
memory with `pytest --memray test_variational_manual_grad.py`. The initial
MPS is seeded (`np.random.seed(0)`), so a tight tolerance is appropriate.
"""
import numpy as np
import pytest
import tenpy.linalg.np_conserved as npc
from tenpy.algorithms.mps_common import OneSiteH
from tenpy.models.spins import SpinChain
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPOEnvironment

from common.model import CHI_VALUES, HEISENBERG_J, L_VALUES, N_GRAD_STEPS, STEP_TIMEOUT_SEC

LEARNING_RATE = 0.1

REFERENCE_ENERGIES = {
    (16, 20): -8.043105653985183,
    (16, 30): -12.402441841968768,
    (16, 50): -21.174989285621006,
    (32, 20): -8.064651077942697,
    (32, 30): -12.487518512563117,
    (32, 50): -21.272032305738456,
    (64, 20): -8.074641390227413,
    (64, 30): -12.502664810104278,
    (64, 50): -21.32802845595387,
}


def run_one(chi, L):
    M = SpinChain(dict(
        L=L, S=0.5, Jx=HEISENBERG_J, Jy=HEISENBERG_J, Jz=HEISENBERG_J,
        bc_MPS="finite", conserve=None,
    ))
    sites = M.lat.mps_sites()
    product_state = (["up", "down"] * (L // 2 + 1))[:L]
    np.random.seed(0)
    psi = MPS.from_random_unitary_evolution(sites, chi, product_state, form="B")
    psi.canonical_form()

    def grad_step():
        env = MPOEnvironment(psi, M.H_MPO, psi)
        energy = None
        R = None
        for i0 in range(L):
            theta = psi.get_theta(i0, n=1)
            if R is not None:
                theta = npc.tensordot(R, theta, axes=["vR", "vL"])
            eff = OneSiteH(env, i0)
            h_theta = eff.matvec(theta)
            norm_sq = npc.inner(theta, theta, axes="range", do_conj=True)
            energy = npc.inner(theta, h_theta, axes="range", do_conj=True) / norm_sq
            grad = 2 * (h_theta - energy * theta)
            new_theta = theta - LEARNING_RATE * grad
            new_theta.ireplace_label("p0", "p")
            if i0 < L - 1:
                combined = new_theta.combine_legs(["vL", "p"], qconj=+1)
                Q, R = npc.qr(combined, inner_labels=["vR", "vL"])
                psi.set_B(i0, Q.split_legs(0), form="A")
            else:
                new_theta /= npc.norm(new_theta)
                psi.set_B(i0, new_theta, form="B")
        psi.canonical_form()
        return energy.real

    energy = None
    for _ in range(N_GRAD_STEPS):
        energy = grad_step()
    return energy


@pytest.mark.timeout(STEP_TIMEOUT_SEC)
@pytest.mark.parametrize("length", L_VALUES)
@pytest.mark.parametrize("chi", CHI_VALUES)
def test_variational_manual_grad_benchmark(benchmark, chi, length):
    energy = benchmark.pedantic(run_one, args=(chi, length), rounds=1, iterations=1)
    benchmark.extra_info["energy"] = float(energy)
    assert float(energy) == pytest.approx(REFERENCE_ENERGIES[(chi, length)], rel=1e-6)


@pytest.mark.limit_memory("40 MB")
def test_variational_manual_grad_memory():
    energy = run_one(16, 20)
    assert float(energy) == pytest.approx(REFERENCE_ENERGIES[(16, 20)], rel=1e-6)
