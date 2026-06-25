"""quimb benchmark, algorithm class 4: variational ground-state search by
gradient descent on the MPS tensors of the 1D Heisenberg chain, using real
automatic differentiation of the Rayleigh quotient

    E(psi) = <psi|H|psi> / <psi|psi>

with respect to every MPS tensor simultaneously. `run_one_jax`/`run_one_torch`
exercise quimb's AD-based optimization on the JAX and PyTorch array backends
respectively.

This is quimb's natural counterpart to the manual analytic gradient used in
the TeNPy (`tenpy_bench/test_variational_manual_grad.py`) and Cytnx
(`cytnx_bench/test_variational_manual_grad.py`) benchmarks: those two
libraries have no autodiff backend, so they evaluate the closed-form
gradient `dE/dA_i* = 2*(H_eff,i(A_i) - E*A_i)` by hand. quimb's MPS/MPO
tensors are plain JAX/PyTorch arrays under the hood (via autoray dispatch),
so here we let the backend's own autodiff differentiate straight through
the full `<psi|H|psi>` and `<psi|psi>` tensor-network contractions instead.

GPU code paths are written for both backends (`device="cuda"` placement)
but cannot be exercised in this environment (no GPU).

Run timing with `pytest --benchmark-only test_variational_ad.py`, memory
with `pytest --memray test_variational_ad.py`. The MPS here is seeded
(`MPS_rand_state(..., seed=0)`), so a tight tolerance is appropriate.
"""
import pytest

import quimb.tensor as qtn

from common.model import CHI_VALUES, HEISENBERG_J, L_VALUES, N_GRAD_STEPS, STEP_TIMEOUT_SEC

LEARNING_RATE = 0.1
DEVICE = "cpu"  # set to "gpu" to exercise the (untested) GPU code paths below

JAX_REFERENCE_ENERGIES = {
    (16, 20): -8.344500541687012,
    (16, 30): -12.314983367919922,
    (16, 50): -21.006853103637695,
    (32, 20): -8.415445327758789,
    (32, 30): -12.765752792358398,
    (32, 50): -21.406782150268555,
    (64, 20): -8.433653831481934,
    (64, 30): -12.814393997192383,
    (64, 50): -21.489011764526367,
}
TORCH_REFERENCE_ENERGIES = {
    (16, 20): -8.34450185868216,
    (16, 30): -12.3149824743681,
    (16, 50): -21.00684729742096,
    (32, 20): -8.415446114804622,
    (32, 30): -12.765754432502138,
    (32, 50): -21.40678314184059,
    (64, 20): -8.433652755086134,
    (64, 30): -12.81439419244885,
    (64, 50): -21.489008825080063,
}


def _build(chi, L):
    psi = qtn.MPS_rand_state(L, bond_dim=chi, dtype="float64", seed=0)
    H = qtn.MPO_ham_heis(L, j=HEISENBERG_J, cyclic=False)
    return psi, H


def run_one_jax(chi, L):
    import jax
    import jax.numpy as jnp

    psi, H = _build(chi, L)
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

    def grad_step(arrays):
        g = grad_fn(arrays)
        new_arrays = [a - LEARNING_RATE * ga for a, ga in zip(arrays, g)]
        # Rescale the whole state by a single global factor derived from
        # <psi|psi>, distributed evenly across all L tensors, rather than
        # normalizing each tensor independently -- the MPS is not in
        # canonical form here, so per-tensor normalization does not keep
        # the contracted <psi|psi> close to 1.
        scale = norm_sq(tuple(new_arrays)) ** (-1.0 / (2 * len(new_arrays)))
        new_arrays = [a * scale for a in new_arrays]
        return tuple(new_arrays)

    for _ in range(N_GRAD_STEPS):
        arrays = grad_step(arrays)
    return float(energy(arrays))


def run_one_torch(chi, L):
    import torch

    psi, H = _build(chi, L)
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
            # <psi|psi>, distributed evenly across all L tensors, rather than
            # normalizing each tensor independently -- the MPS is not in
            # canonical form here, so per-tensor normalization does not keep
            # the contracted <psi|psi> close to 1.
            scale = norm_sq(new_arrays) ** (-1.0 / (2 * len(new_arrays)))
            new_arrays = [(a * scale).clone().requires_grad_(True) for a in new_arrays]
        return new_arrays

    for _ in range(N_GRAD_STEPS):
        arrays = grad_step(arrays)
    with torch.no_grad():
        return float(energy(arrays))


@pytest.mark.timeout(STEP_TIMEOUT_SEC)
@pytest.mark.parametrize("length", L_VALUES)
@pytest.mark.parametrize("chi", CHI_VALUES)
def test_variational_ad_jax_benchmark(benchmark, chi, length):
    energy = benchmark.pedantic(run_one_jax, args=(chi, length), rounds=1, iterations=1)
    benchmark.extra_info["energy"] = energy
    assert energy == pytest.approx(JAX_REFERENCE_ENERGIES[(chi, length)], rel=1e-4)


@pytest.mark.limit_memory("800 MB")
def test_variational_ad_jax_memory():
    energy = run_one_jax(16, 20)
    assert energy == pytest.approx(JAX_REFERENCE_ENERGIES[(16, 20)], rel=1e-4)


@pytest.mark.timeout(STEP_TIMEOUT_SEC)
@pytest.mark.parametrize("length", L_VALUES)
@pytest.mark.parametrize("chi", CHI_VALUES)
def test_variational_ad_torch_benchmark(benchmark, chi, length):
    energy = benchmark.pedantic(run_one_torch, args=(chi, length), rounds=1, iterations=1)
    benchmark.extra_info["energy"] = energy
    assert energy == pytest.approx(TORCH_REFERENCE_ENERGIES[(chi, length)], rel=1e-4)


@pytest.mark.limit_memory("100 MB")
def test_variational_ad_torch_memory():
    energy = run_one_torch(16, 20)
    assert energy == pytest.approx(TORCH_REFERENCE_ENERGIES[(16, 20)], rel=1e-4)
