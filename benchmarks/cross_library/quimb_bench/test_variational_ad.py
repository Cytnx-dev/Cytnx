"""quimb benchmark, algorithm class 4: variational ground-state search by
gradient descent on the MPS tensors of the 1D Heisenberg chain, using real
automatic differentiation of the Rayleigh quotient

    E(psi) = <psi|H|psi> / <psi|psi>

with respect to every MPS tensor simultaneously: one gradient step is
taken on every MPS tensor at once (no orthogonality center, no per-step
canonicalization), followed by a single global rescale derived from the
new `<psi|psi>` and distributed evenly across all num_sites tensors.
`run_one_jax`/`run_one_torch` exercise quimb's AD-based optimization on the
JAX and PyTorch array backends respectively.

TeNPy (`tenpy_bench/test_variational_manual_grad.py`) and Cytnx
(`cytnx_bench/test_variational_manual_grad.py`) run this same whole-network
update; having no autodiff backend, they evaluate the closed-form gradient

    dE/dA_i* = (2 / den) * (H_eff,i(A_i) - E * N_eff,i(A_i))

by hand instead, where `den = <psi|psi>` and `N_eff,i` is the
no-MPO analogue of `H_eff,i` (needed because the tensors away from site i
are not isometric under this update, unlike in a one-site sweep). quimb's
MPS/MPO tensors are plain JAX/PyTorch arrays under the hood (via autoray
dispatch), so here we let the backend's own autodiff differentiate straight
through the full `<psi|H|psi>` and `<psi|psi>` contractions instead of
deriving that gradient by hand.

This whole-network update moves the state far less per iteration than a
one-site sweep does, so it needs both a larger learning rate and many more
iterations to reach a comparable energy neighborhood; `LEARNING_RATE` and
the local `_n_grad_steps(num_sites)` helper (scaling with `num_sites`,
since a longer chain needs proportionally more whole-state updates to
converge as far) are shared with the TeNPy/Cytnx benchmarks. They were
picked by checking, at every (bond_dim, num_sites) grid point, that the
resulting energy lands within the
`rel=2e-2` tolerance used below while comfortably inside the timeout. Since
this update is a much weaker optimizer than a one-site sweep, its converged
energy is sensitive to each library's own initial-state construction and
RNG, so the `rel=2e-2` tolerance is wider than the tight per-library
tolerances used elsewhere in this suite -- it is not expected to shrink as
the manual-gradient benchmarks are made more precise.

GPU code paths are written for both backends (`device="cuda"` placement)
but cannot be exercised in this environment (no GPU).

Run timing with `pytest --benchmark-only test_variational_ad.py`, memory
with `pytest --memray test_variational_ad.py`. The MPS here is seeded
(`MPS_rand_state(..., seed=0)`).
"""
import pytest

import quimb.tensor as qtn

from common.model import BOND_DIM_VALUES, HEISENBERG_J, NUM_SITES_VALUES, GRID_POINT_TIMEOUT_SEC

LEARNING_RATE = 0.5
DEVICE = "cpu"  # set to "gpu" to exercise the (untested) GPU code paths below

JAX_REFERENCE_ENERGIES = {
    (16, 20): -8.67426586151123,
    (16, 30): -13.085174560546875,
    (16, 50): -21.572669982910156,
    (32, 20): -8.659192085266113,
    (32, 30): -13.020613670349121,
    (32, 50): -21.64177703857422,
    (64, 20): -8.671019554138184,
    (64, 30): -13.056256294250488,
    (64, 50): -21.65955352783203,
}
TORCH_REFERENCE_ENERGIES = {
    (16, 20): -8.674265280037904,
    (16, 30): -13.085195694053375,
    (16, 50): -21.606009355825424,
    (32, 20): -8.659190668103744,
    (32, 30): -13.020607976110528,
    (32, 50): -21.639568826632235,
    (64, 20): -8.671020853476218,
    (64, 30): -13.056264576677071,
    (64, 50): -21.589291140405937,
}


def _build(bond_dim, num_sites):
    psi = qtn.MPS_rand_state(num_sites, bond_dim=bond_dim, dtype="float64", seed=0)
    H = qtn.MPO_ham_heis(num_sites, j=HEISENBERG_J, cyclic=False)
    return psi, H


def _n_grad_steps(num_sites):
    return 8 * num_sites


def run_one_jax(bond_dim, num_sites):
    import jax
    import jax.numpy as jnp

    psi, H = _build(bond_dim, num_sites)
    if DEVICE == "gpu":
        device = jax.devices("gpu")[0]
    else:
        device = jax.devices("cpu")[0]
    arrays = tuple(jax.device_put(jnp.asarray(a), device) for a in psi.arrays)
    H.apply_to_arrays(lambda x: jax.device_put(jnp.asarray(x), device))

    def energy(arrays):
        p = psi.copy()
        for i, a in enumerate(arrays):
            p[i].modify(data=a)
        num = p.H @ (H.apply(p))
        den = p.H @ p
        return jnp.real(num / den)

    def norm_sq(arrays):
        p = psi.copy()
        for i, a in enumerate(arrays):
            p[i].modify(data=a)
        return jnp.real(p.H @ p)

    grad_fn = jax.jit(jax.grad(energy)) if DEVICE == "cpu" else jax.grad(energy)
    norm_sq_fn = jax.jit(norm_sq) if DEVICE == "cpu" else norm_sq

    def grad_step(arrays):
        g = grad_fn(arrays)
        new_arrays = [a - LEARNING_RATE * ga for a, ga in zip(arrays, g)]
        # Rescale the whole state by a single global factor derived from
        # <psi|psi>, distributed evenly across all num_sites tensors, rather than
        # normalizing each tensor independently -- the MPS is not in
        # canonical form here, so per-tensor normalization does not keep
        # the contracted <psi|psi> close to 1.
        scale = norm_sq_fn(tuple(new_arrays)) ** (-1.0 / (2 * len(new_arrays)))
        new_arrays = [a * scale for a in new_arrays]
        return tuple(new_arrays)

    for _ in range(_n_grad_steps(num_sites)):
        arrays = grad_step(arrays)
    return float(energy(arrays))


def run_one_torch(bond_dim, num_sites):
    import torch

    psi, H = _build(bond_dim, num_sites)
    torch_device = "cuda" if DEVICE == "gpu" else "cpu"
    arrays = [
        torch.as_tensor(a, dtype=torch.float64, device=torch_device).clone().requires_grad_(True)
        for a in psi.arrays
    ]
    H.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float64, device=torch_device))

    def energy(arrays):
        p = psi.copy()
        for i, a in enumerate(arrays):
            p[i].modify(data=a)
        num = p.H @ (H.apply(p))
        den = p.H @ p
        e = num / den
        return torch.real(e) if torch.is_complex(e) else e

    def norm_sq(arrays):
        p = psi.copy()
        for i, a in enumerate(arrays):
            p[i].modify(data=a)
        return p.H @ p

    def grad_step(arrays):
        for a in arrays:
            if a.grad is not None:
                a.grad = None
        e = energy(arrays)
        e.backward()
        new_arrays = []
        with torch.no_grad():
            for a in arrays:
                a_new = a - LEARNING_RATE * a.grad
                new_arrays.append(a_new)
            # Rescale the whole state by a single global factor derived from
            # <psi|psi>, distributed evenly across all num_sites tensors, rather than
            # normalizing each tensor independently -- the MPS is not in
            # canonical form here, so per-tensor normalization does not keep
            # the contracted <psi|psi> close to 1.
            scale = norm_sq(new_arrays) ** (-1.0 / (2 * len(new_arrays)))
            new_arrays = [(a * scale).clone().requires_grad_(True) for a in new_arrays]
        return new_arrays

    for _ in range(_n_grad_steps(num_sites)):
        arrays = grad_step(arrays)
    with torch.no_grad():
        return float(energy(arrays))


@pytest.mark.timeout(GRID_POINT_TIMEOUT_SEC)
@pytest.mark.parametrize("num_sites", NUM_SITES_VALUES)
@pytest.mark.parametrize("bond_dim", BOND_DIM_VALUES)
def test_variational_ad_jax_benchmark(benchmark, bond_dim, num_sites):
    energy = benchmark.pedantic(run_one_jax, args=(bond_dim, num_sites), rounds=1, iterations=1)
    benchmark.extra_info["energy"] = energy
    assert energy == pytest.approx(JAX_REFERENCE_ENERGIES[(bond_dim, num_sites)], rel=2e-2)


@pytest.mark.cytnx_memory
@pytest.mark.limit_memory("800 MB")
@pytest.mark.parametrize("num_sites", NUM_SITES_VALUES)
@pytest.mark.parametrize("bond_dim", BOND_DIM_VALUES)
def test_variational_ad_jax_memory(bond_dim, num_sites):
    energy = run_one_jax(bond_dim, num_sites)
    assert energy == pytest.approx(JAX_REFERENCE_ENERGIES[(bond_dim, num_sites)], rel=2e-2)


@pytest.mark.timeout(GRID_POINT_TIMEOUT_SEC)
@pytest.mark.parametrize("num_sites", NUM_SITES_VALUES)
@pytest.mark.parametrize("bond_dim", BOND_DIM_VALUES)
def test_variational_ad_torch_benchmark(benchmark, bond_dim, num_sites):
    energy = benchmark.pedantic(run_one_torch, args=(bond_dim, num_sites), rounds=1, iterations=1)
    benchmark.extra_info["energy"] = energy
    assert energy == pytest.approx(TORCH_REFERENCE_ENERGIES[(bond_dim, num_sites)], rel=2e-2)


@pytest.mark.cytnx_memory
@pytest.mark.limit_memory("400 MB")
@pytest.mark.parametrize("num_sites", NUM_SITES_VALUES)
@pytest.mark.parametrize("bond_dim", BOND_DIM_VALUES)
def test_variational_ad_torch_memory(bond_dim, num_sites):
    energy = run_one_torch(bond_dim, num_sites)
    assert energy == pytest.approx(TORCH_REFERENCE_ENERGIES[(bond_dim, num_sites)], rel=2e-2)
