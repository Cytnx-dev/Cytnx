"""quimb benchmark, algorithm class 4: variational ground-state search by
gradient descent on the MPS tensors of the 1D Heisenberg chain, using real
automatic differentiation in place of a hand-derived gradient.

This runs the *same* one-site sweep algorithm as the TeNPy
(`tenpy_bench/test_variational_manual_grad.py`) and Cytnx
(`cytnx_bench/test_variational_manual_grad.py`) benchmarks: at each site i,
build the effective one-site Hamiltonian H_eff,i by contracting the MPO
with the left/right boundary environments, take a gradient-descent step on
the Rayleigh quotient E(theta) = <theta|H_eff,i|theta> / <theta|theta>,
normalize, and push the gauge forward (left-canonicalize the just-updated
site via SVD so the orthogonality center advances to site i+1); after a
full left-to-right sweep, restore the right-canonical gauge with a chain
of SVDs. The only difference from the TeNPy/Cytnx benchmarks is how the
per-site gradient is obtained: instead of the closed form
dE/dA_i* = 2*(H_eff,i(A_i) - E*A_i), `jax.grad`/`torch.autograd` differentiate
straight through the same `local_energy` contraction.

quimb's own MPS/MPO classes are not used here. quimb has no API for
holding a sweep at a single-site orthogonality center while differentiating
only the local effective Hamiltonian -- its autodiff support operates on
whole-network contractions, which is a different algorithm (gradient
descent on every tensor simultaneously, not a one-site sweep), so building
the boundary-environment and effective-Hamiltonian contractions directly
out of plain JAX/PyTorch arrays mirrors the TeNPy benchmark's rationale for
bypassing `tenpy.networks.mps.MPS`/`MPOEnvironment` in the same situation.

The MPO (open boundary, bond dimension 5) and the per-site i.i.d.-normal
initial MPS (same per-site bond-dimension formula and per-site
`np.random.RandomState(seed)` draw, right-canonicalized via a chain of
SVDs before the first sweep) are built identically to the TeNPy benchmark,
so that the only axis of variation across TeNPy/Cytnx/quimb is the
gradient-computation method, not the optimization trajectory.

JAX runs with `jax_enable_x64` so its contractions use the same float64
precision as the PyTorch path and the TeNPy/Cytnx manual gradient, rather
than JAX's float32 default silently introducing a precision-driven
mismatch unrelated to the algorithm itself.

GPU code paths are written for both backends (`device="cuda"` placement)
but cannot be exercised in this environment (no GPU).

Run timing with `pytest --benchmark-only test_variational_ad.py`, memory
with `pytest --memray test_variational_ad.py`. The initial MPS is seeded
(per-site `np.random.RandomState(seed)`), so a tight tolerance is
appropriate.
"""
import numpy as np
import pytest

from common.model import CHI_VALUES, HEISENBERG_J, L_VALUES, N_GRAD_STEPS, STEP_TIMEOUT_SEC

LEARNING_RATE = 0.1
DEVICE = "cpu"  # set to "gpu" to exercise the (untested) GPU code paths below

JAX_REFERENCE_ENERGIES = {
    (16, 20): -8.68246845559463,
    (16, 30): -13.111313297814393,
    (16, 50): -21.97157216944336,
    (32, 20): -8.682473317775248,
    (32, 30): -13.111355489545577,
    (32, 50): -21.97210625282108,
    (64, 20): -8.682473333622669,
    (64, 30): -13.111355749012468,
    (64, 50): -21.972110271827862,
}
TORCH_REFERENCE_ENERGIES = {
    (16, 20): -8.682468455594627,
    (16, 30): -13.111313297814295,
    (16, 50): -21.971572169443302,
    (32, 20): -8.682473317775242,
    (32, 30): -13.111355489545595,
    (32, 50): -21.972106252821057,
    (64, 20): -8.682473333622703,
    (64, 30): -13.111355749012533,
    (64, 50): -21.97211027182776,
}


def _build_mpo(J):
    d, D = 2, 5
    Sp = np.zeros((d, d)); Sp[0, 1] = 1.0
    Sm = np.zeros((d, d)); Sm[1, 0] = 1.0
    Sz = np.zeros((d, d)); Sz[0, 0] = 0.5; Sz[1, 1] = -0.5
    eye = np.eye(d)
    M = np.zeros((D, D, d, d))
    M[0, 0] = eye
    M[D - 1, D - 1] = eye
    M[0, 1] = Sp
    M[0, 2] = Sm
    M[0, 3] = Sz
    M[1, D - 1] = (J / 2.0) * Sm
    M[2, D - 1] = (J / 2.0) * Sp
    M[3, D - 1] = J * Sz
    L0 = np.zeros((D, 1, 1)); L0[0, 0, 0] = 1.0
    R0 = np.zeros((D, 1, 1)); R0[D - 1, 0, 0] = 1.0
    return M, L0, R0


def _build_mps(L, chi, d=2):
    A = [None] * L
    A[0] = np.random.RandomState(0).normal(size=(1, d, min(chi, d)))
    for k in range(1, L):
        dim1 = A[k - 1].shape[2]
        dim3 = min(min(chi, dim1 * d), d ** (L - k - 1))
        A[k] = np.random.RandomState(k).normal(size=(dim1, d, dim3))
    _canonicalize_right(A, L)
    return A


def _canonicalize_right(A, L):
    for p in range(L - 1, 0, -1):
        dim1, d, dim3 = A[p].shape
        mat = A[p].reshape(dim1, d * dim3)
        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        A[p] = vh.reshape(-1, d, dim3)
        A[p - 1] = np.einsum('abc,cd->abd', A[p - 1], u * s[None, :])
    A[0] = A[0] / np.linalg.norm(A[0])


def run_one_jax(chi, L):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    if DEVICE == "gpu":
        device = jax.devices("gpu")[0]
    else:
        device = jax.devices("cpu")[0]

    M_np, L0_np, R0_np = _build_mpo(HEISENBERG_J)
    M = jax.device_put(jnp.asarray(M_np), device)
    L0 = jax.device_put(jnp.asarray(L0_np), device)
    R0 = jax.device_put(jnp.asarray(R0_np), device)
    A = [jax.device_put(jnp.asarray(a), device) for a in _build_mps(L, chi)]

    def update_L(LP, A_i, M_i):
        return jnp.einsum('abc,bfg,adef,ceh->dgh', LP, A_i, M_i, A_i.conj())

    def update_R(RP, A_i, M_i):
        return jnp.einsum('bfg,adef,dgh,ceh->abc', A_i, M_i, RP, A_i.conj())

    def h_eff(theta, LP, RP, M_i):
        return jnp.einsum('abc,adef,bfg,dgh->ceh', LP, M_i, theta, RP)

    def local_energy(theta, LP, RP, M_i):
        ht = h_eff(theta, LP, RP, M_i)
        norm_sq = jnp.sum(theta * theta)
        numer = jnp.sum(theta * ht)
        return numer / norm_sq

    energy_fn = jax.jit(local_energy) if DEVICE == "cpu" else local_energy
    grad_fn = jax.jit(jax.grad(local_energy)) if DEVICE == "cpu" else jax.grad(local_energy)

    def grad_step(A):
        R_env = [None] * (L + 1)
        R_env[L] = R0
        for p in range(L - 1, 0, -1):
            R_env[p] = update_R(R_env[p + 1], A[p], M)
        L_env = L0
        energy = None
        for p in range(L):
            theta = A[p]
            energy = energy_fn(theta, L_env, R_env[p + 1], M)
            grad = grad_fn(theta, L_env, R_env[p + 1], M)
            new_theta = theta - LEARNING_RATE * grad
            new_theta = new_theta / jnp.linalg.norm(new_theta)
            if p < L - 1:
                dim1, d, dim3 = new_theta.shape
                mat = new_theta.reshape(dim1 * d, dim3)
                u, s, vh = jnp.linalg.svd(mat, full_matrices=False)
                k = s.shape[0]
                A[p] = u.reshape(dim1, d, k)
                sv = s[:, None] * vh
                A[p + 1] = jnp.einsum('kd,dpr->kpr', sv, A[p + 1])
            else:
                A[p] = new_theta
            L_env = update_L(L_env, A[p], M)
        A_np = [np.array(a) for a in A]
        _canonicalize_right(A_np, L)
        A[:] = [jax.device_put(jnp.asarray(a), device) for a in A_np]
        return energy

    energy = None
    for _ in range(N_GRAD_STEPS):
        energy = grad_step(A)
    return float(energy)


def run_one_torch(chi, L):
    import torch

    torch_device = "cuda" if DEVICE == "gpu" else "cpu"
    M_np, L0_np, R0_np = _build_mpo(HEISENBERG_J)
    M = torch.as_tensor(M_np, dtype=torch.float64, device=torch_device)
    L0 = torch.as_tensor(L0_np, dtype=torch.float64, device=torch_device)
    R0 = torch.as_tensor(R0_np, dtype=torch.float64, device=torch_device)
    A = [torch.as_tensor(a, dtype=torch.float64, device=torch_device) for a in _build_mps(L, chi)]

    def update_L(LP, A_i, M_i):
        return torch.einsum('abc,bfg,adef,ceh->dgh', LP, A_i, M_i, A_i.conj())

    def update_R(RP, A_i, M_i):
        return torch.einsum('bfg,adef,dgh,ceh->abc', A_i, M_i, RP, A_i.conj())

    def h_eff(theta, LP, RP, M_i):
        return torch.einsum('abc,adef,bfg,dgh->ceh', LP, M_i, theta, RP)

    def local_energy(theta, LP, RP, M_i):
        ht = h_eff(theta, LP, RP, M_i)
        norm_sq = torch.sum(theta * theta)
        numer = torch.sum(theta * ht)
        return numer / norm_sq

    def grad_step(A):
        R_env = [None] * (L + 1)
        R_env[L] = R0
        for p in range(L - 1, 0, -1):
            R_env[p] = update_R(R_env[p + 1], A[p], M)
        L_env = L0
        energy = None
        for p in range(L):
            theta = A[p].clone().requires_grad_(True)
            e = local_energy(theta, L_env, R_env[p + 1], M)
            e.backward()
            with torch.no_grad():
                new_theta = theta - LEARNING_RATE * theta.grad
                new_theta = new_theta / torch.linalg.norm(new_theta)
            energy = e.detach()
            if p < L - 1:
                dim1, d, dim3 = new_theta.shape
                mat = new_theta.reshape(dim1 * d, dim3)
                u, s, vh = torch.linalg.svd(mat, full_matrices=False)
                k = s.shape[0]
                A[p] = u.reshape(dim1, d, k)
                sv = s[:, None] * vh
                A[p + 1] = torch.einsum('kd,dpr->kpr', sv, A[p + 1])
            else:
                A[p] = new_theta
            L_env = update_L(L_env, A[p], M)
        A_np = [a.detach().numpy() for a in A]
        _canonicalize_right(A_np, L)
        A[:] = [torch.as_tensor(a, dtype=torch.float64, device=torch_device) for a in A_np]
        return energy

    energy = None
    for _ in range(N_GRAD_STEPS):
        energy = grad_step(A)
    return float(energy)


@pytest.mark.timeout(STEP_TIMEOUT_SEC)
@pytest.mark.parametrize("length", L_VALUES)
@pytest.mark.parametrize("chi", CHI_VALUES)
def test_variational_ad_jax_benchmark(benchmark, chi, length):
    energy = benchmark.pedantic(run_one_jax, args=(chi, length), rounds=1, iterations=1)
    benchmark.extra_info["energy"] = energy
    assert energy == pytest.approx(JAX_REFERENCE_ENERGIES[(chi, length)], rel=1e-6)


@pytest.mark.limit_memory("100 MB")
def test_variational_ad_jax_memory():
    energy = run_one_jax(16, 20)
    assert energy == pytest.approx(JAX_REFERENCE_ENERGIES[(16, 20)], rel=1e-6)


@pytest.mark.timeout(STEP_TIMEOUT_SEC)
@pytest.mark.parametrize("length", L_VALUES)
@pytest.mark.parametrize("chi", CHI_VALUES)
def test_variational_ad_torch_benchmark(benchmark, chi, length):
    energy = benchmark.pedantic(run_one_torch, args=(chi, length), rounds=1, iterations=1)
    benchmark.extra_info["energy"] = energy
    assert energy == pytest.approx(TORCH_REFERENCE_ENERGIES[(chi, length)], rel=1e-6)


@pytest.mark.limit_memory("100 MB")
def test_variational_ad_torch_memory():
    energy = run_one_torch(16, 20)
    assert energy == pytest.approx(TORCH_REFERENCE_ENERGIES[(16, 20)], rel=1e-6)
