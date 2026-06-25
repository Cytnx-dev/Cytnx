"""Cytnx benchmark, algorithm class 4: variational ground-state search by
gradient descent on the MPS tensors of the 1D Heisenberg chain, with a
hand-derived (manual) gradient instead of automatic differentiation.

Cytnx has no autodiff backend, so the gradient of the Rayleigh quotient
    E(psi) = <psi|H|psi> / <psi|psi>
with respect to a single MPS tensor A_i (holding all other tensors fixed)
is computed analytically rather than via backprop:

    dE/dA_i* = 2 * (H_eff,i(A_i) - E * A_i)

where H_eff,i is the effective one-site Hamiltonian obtained by contracting
the MPO with the left/right boundary environments around site i. H_eff,i is
built here from Cytnx's own `UniTensor`/`Network`/`Contract` primitives,
reusing the same bond-dimension-5 Heisenberg MPO and L/R environment-update
`Network` definitions as `test_dmrg_dense.py` (those networks already
operate on a single MPS tensor plus its conjugate, so they apply unchanged
to a one-site sweep). Right environments for not-yet-visited sites are
computed once per gradient sweep and left environments are updated
incrementally as the sweep passes each site -- this is a *contraction*, not
automatic differentiation, and is written independently of the closed-form
gradient used in the TeNPy (`tenpy_bench/test_variational_manual_grad.py`,
`np_conserved` contractions) and quimb (`quimb_bench/test_variational_ad.py`,
real autodiff) benchmarks.

CPU and GPU code paths are both written; the GPU path moves every MPS/MPO
UniTensor to `cytnx.Device.cuda` before the gradient sweep. It cannot be
exercised in this environment (no GPU).

Run timing with `pytest --benchmark-only test_variational_manual_grad.py`,
memory with `pytest --memray test_variational_manual_grad.py`.
"""
import pytest

import cytnx

from common.model import CHI_VALUES, HEISENBERG_J, L_VALUES, N_GRAD_STEPS, STEP_TIMEOUT_SEC

DEVICE = "cpu"  # set to "gpu" to exercise the (untested) GPU code path below
LEARNING_RATE = 0.1

REFERENCE_ENERGIES = {
    (16, 20): -8.682468455146315,
    (16, 30): -13.111313399023222,
    (16, 50): -21.971715188684332,
    (32, 20): -8.682473317623886,
    (32, 30): -13.111355507543065,
    (32, 50): -21.972106247315416,
    (64, 20): -8.68247333435864,
    (64, 30): -13.111355758459972,
    (64, 50): -21.97211027152718,
}

_L_UPDATE_NET = ["L: -2,-1,-3", "A: -1,-4,1", "M: -2,0,-4,-5", "A_Conj: -3,-5,2", "TOUT: 0,1,2"]
_R_UPDATE_NET = ["R: -2,-1,-3", "B: 1,-4,-1", "M: 0,-2,-4,-5", "B_Conj: 2,-5,-3", "TOUT: 0,1,2"]
_HEFF_NET = ["psi: -1,-2,-3", "L: -4,-1,0", "R: -5,-3,2", "M: -4,-5,-2,1", "TOUT: 0,1;2"]


def _build_mpo(J, device):
    d = 2
    D = 5
    Sp = cytnx.zeros([d, d])
    Sp[0, 1] = 1.0  # S+: |down> -> |up>
    Sm = cytnx.zeros([d, d])
    Sm[1, 0] = 1.0  # S-: |up> -> |down>
    Sz = cytnx.zeros([d, d])
    Sz[0, 0] = 0.5
    Sz[1, 1] = -0.5
    eye = cytnx.eye(d)

    M = cytnx.zeros([D, D, d, d])
    M[0, 0] = eye
    M[D - 1, D - 1] = eye
    M[0, 1] = Sp
    M[0, 2] = Sm
    M[0, 3] = Sz
    M[1, D - 1] = (J / 2.0) * Sm
    M[2, D - 1] = (J / 2.0) * Sp
    M[3, D - 1] = J * Sz
    M = cytnx.UniTensor(M, 0).set_name("MPO")

    L0 = cytnx.UniTensor.zeros([D, 1, 1]).set_rowrank_(0).set_name("L0")
    R0 = cytnx.UniTensor.zeros([D, 1, 1]).set_rowrank_(0).set_name("R0")
    L0[0, 0, 0] = 1.0
    R0[D - 1, 0, 0] = 1.0
    if device == "gpu":
        M = M.to(cytnx.Device.cuda)
        L0 = L0.to(cytnx.Device.cuda)
        R0 = R0.to(cytnx.Device.cuda)
    return M, L0, R0


def _build_mps(L, chi, device):
    d = 2
    A = [None] * L
    lbls = [[str(2 * k), str(2 * k + 1), str(2 * k + 2)] for k in range(L)]
    A[0] = cytnx.UniTensor.normal([1, d, min(chi, d)], 0., 1., seed=0).set_rowrank_(2)
    A[0].relabel_(lbls[0]).set_name("A0")
    for k in range(1, L):
        dim1 = A[k - 1].shape()[2]
        dim3 = min(min(chi, dim1 * d), d ** (L - k - 1))
        A[k] = cytnx.UniTensor.normal([dim1, d, dim3], 0., 1., seed=k).set_rowrank_(2)
        A[k].relabel_(lbls[k]).set_name(f"A{k}")
    _canonicalize_right(A, lbls, L)
    if device == "gpu":
        A = [a.to(cytnx.Device.cuda) for a in A]
    return A, lbls


def _canonicalize_right(A, lbls, L):
    for p in range(L - 1, 0, -1):
        A[p].set_rowrank_(1)
        s, u, vt = cytnx.linalg.Gesvd(A[p])
        A[p] = vt
        A[p - 1] = cytnx.Contract(A[p - 1], cytnx.Contract(u, s))
        A[p].relabel_(lbls[p]).set_name(f"A{p}")
        A[p - 1].relabel_(lbls[p - 1]).set_name(f"A{p-1}")
    A[0] = A[0] / A[0].Norm().item()
    A[0].relabel_(lbls[0]).set_name("A0")


def _update_L(L_env, A_new, M):
    net = cytnx.Network()
    net.FromString(_L_UPDATE_NET)
    net.PutUniTensors(["L", "A", "A_Conj", "M"], [L_env, A_new, A_new.Dagger().permute_(A_new.labels()), M])
    return net.Launch()


def _update_R(R_env, B_new, M):
    net = cytnx.Network()
    net.FromString(_R_UPDATE_NET)
    net.PutUniTensors(["R", "B", "M", "B_Conj"], [R_env, B_new, M, B_new.Dagger().permute_(B_new.labels())])
    return net.Launch()


def _h_eff(theta, L_env, R_env, M):
    net = cytnx.Network()
    net.FromString(_HEFF_NET)
    net.PutUniTensors(["psi", "L", "R", "M"], [theta, L_env, R_env, M])
    out = net.Launch()
    out.relabel_(theta.labels())
    return out


def run_one(chi, L):
    device = "gpu" if DEVICE == "gpu" else "cpu"
    M, L0, R0 = _build_mpo(HEISENBERG_J, device)
    A, lbls = _build_mps(L, chi, device)

    def grad_step():
        R_env = [None] * (L + 1)
        R_env[L] = R0
        for p in range(L - 1, 0, -1):
            R_env[p] = _update_R(R_env[p + 1], A[p], M)

        L_env = L0
        energy = None
        for p in range(L):
            theta = A[p]
            h_theta = _h_eff(theta, L_env, R_env[p + 1], M)
            theta_dag = theta.Dagger().permute_(theta.labels())
            norm_sq = cytnx.Contract(theta_dag, theta).item()
            energy = cytnx.Contract(theta_dag, h_theta).item() / norm_sq
            grad = 2 * (h_theta - energy * theta)
            new_theta = theta - LEARNING_RATE * grad
            new_theta = new_theta / new_theta.Norm().item()
            # Cytnx arithmetic ops reset UniTensor labels to the default
            # ['0','1',...] sequence rather than preserving theta's labels,
            # so new_theta must be relabeled back to the site's real bond
            # names before any Gesvd split -- otherwise the split-off bond
            # leg carries the wrong label and silently fails to contract
            # with A[p+1]'s matching leg.
            new_theta.relabel_(lbls[p])
            if p < L - 1:
                # Push the orthogonality center forward (left-canonicalize the
                # just-updated site) so that H_eff at the next site is built
                # against a properly left-orthonormal left environment, as the
                # Rayleigh-quotient shortcut `energy = <theta|H_eff|theta>`
                # requires.
                new_theta.set_rowrank_(2)
                s, A[p], vt = cytnx.linalg.Gesvd(new_theta)
                A[p + 1] = cytnx.Contract(cytnx.Contract(s, vt), A[p + 1])
                A[p + 1].relabel_(lbls[p + 1]).set_name(f"A{p+1}")
            else:
                A[p] = new_theta
            A[p].set_name(f"A{p}").relabel_(lbls[p])
            L_env = _update_L(L_env, A[p], M)
        # Restore the right-canonical gauge (all sites right-orthonormal
        # except A[0]) so the next sweep's R_env precomputation and
        # Rayleigh-quotient shortcut remain valid.
        _canonicalize_right(A, lbls, L)
        return energy

    energy = None
    for _ in range(N_GRAD_STEPS):
        energy = grad_step()
    return energy


@pytest.mark.timeout(STEP_TIMEOUT_SEC)
@pytest.mark.parametrize("length", L_VALUES)
@pytest.mark.parametrize("chi", CHI_VALUES)
def test_variational_manual_grad_benchmark(benchmark, chi, length):
    energy = benchmark.pedantic(run_one, args=(chi, length), rounds=1, iterations=1)
    benchmark.extra_info["energy"] = energy
    assert energy == pytest.approx(REFERENCE_ENERGIES[(chi, length)], rel=1e-6)


@pytest.mark.limit_memory("20 MB")
def test_variational_manual_grad_memory():
    energy = run_one(16, 20)
    assert energy == pytest.approx(REFERENCE_ENERGIES[(16, 20)], rel=1e-6)
