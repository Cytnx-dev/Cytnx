"""Cytnx benchmark, algorithm class 2: real-time evolution of a 1D
transverse-field Ising chain after a field quench, using a hand-rolled TEBD
sweep built from Cytnx's own `UniTensor`/`Contract`/`Svd_truncate` API (Cytnx
has no built-in TEBD engine, unlike TeNPy/quimb).

Each Trotter layer applies the two-site gate exp(-i*dt*h_bond) to every bond
in a single strictly sequential left-to-right sweep (bond 0, then 1, ...,
then L-2), absorbing the post-SVD singular-value tensor into the
not-yet-visited (right) neighbor every time so the orthogonality center
moves forward with the sweep -- a fixed absorption side regardless of sweep
direction breaks the canonical gauge between already-updated and
not-yet-updated tensors. The on-site field term -hx*Sx is split between the
two bond gates that touch each site (half-weight on interior bonds, full
weight on the two chain-boundary bonds) so that summing the per-bond gates'
field contributions over a full sweep reproduces -hx*sum(Sx_i) exactly once
per site rather than twice for interior sites. The initial state is the
all-spin-down product state, evolved directly under the post-quench
Hamiltonian (matching the `quimb_bench/tebd.py` convention).

CPU and GPU code paths are both written; the GPU path moves every MPS
UniTensor and the gate to `cytnx.Device.cuda` before stepping. It cannot be
exercised in this environment (no GPU).

Run timing with `pytest --benchmark-only test_tebd.py`, memory with
`pytest --memray test_tebd.py`.
"""
import pytest

import cytnx

from common.model import BOND_DIM_VALUES, NUM_SITES_VALUES, GRID_POINT_TIMEOUT_SEC, TFIM_DT, TFIM_HX_FINAL, TFIM_J, TFIM_N_STEPS

DEVICE = "cpu"  # set to "gpu" to exercise the (untested) GPU code path below

REFERENCE_ENERGIES = {
    (16, 20): -19.000359069981286,
    (16, 30): -28.999909822622687,
    (16, 50): -48.9994338138508,
    (32, 20): -19.000634976726793,
    (32, 30): -29.00076260540032,
    (32, 50): -49.00137490785111,
    (64, 20): -19.0005361684381,
    (64, 30): -29.000882075567542,
    (64, 50): -49.001545912461594,
}


def _build_gate(J, hx, dt, w_left, w_right, device):
    Sz = cytnx.physics.pauli("z", device=device).real()
    Sx = cytnx.physics.pauli("x", device=device).real()
    I = cytnx.eye(2, device=device)
    TFterm = w_left * cytnx.linalg.Kron(Sx, I) + w_right * cytnx.linalg.Kron(I, Sx)
    ZZterm = cytnx.linalg.Kron(Sz, Sz)
    H = -hx * TFterm - J * ZZterm
    eH = cytnx.linalg.ExpH(H, -1j * dt)
    eH.reshape_(2, 2, 2, 2)
    return cytnx.UniTensor(eH)


def _build_gates(L, J, hx, dt, device):
    gates = []
    for p in range(L - 1):
        w_left = 1.0 if p == 0 else 0.5
        w_right = 1.0 if p == L - 2 else 0.5
        gates.append(_build_gate(J, hx, dt, w_left, w_right, device))
    return gates


def _build_mps(L, chi, device):
    d = 2
    A = [None] * L
    lbls = [[str(2 * k), str(2 * k + 1), str(2 * k + 2)] for k in range(L)]
    for k in range(L):
        A[k] = cytnx.UniTensor.zeros([1, d, 1], device=device).set_rowrank_(2).relabel_(lbls[k]).set_name(f"A{k}")
        A[k].set_elem([0, 0, 0], 1.0)
    return A, lbls


def _build_mpo(J, hx, device):
    D = 3
    Sz = cytnx.physics.pauli("z", device=device).real()
    Sx = cytnx.physics.pauli("x", device=device).real()
    eye = cytnx.eye(2, device=device)

    M = cytnx.zeros([D, D, 2, 2], device=device)
    M[0, 0] = eye
    M[0, 1] = Sz
    M[0, 2] = -hx * Sx
    M[1, 2] = -J * Sz
    M[D - 1, D - 1] = eye
    M = cytnx.UniTensor(M, 0).set_name("MPO")

    L0 = cytnx.UniTensor.zeros([D, 1, 1], device=device).set_rowrank_(0).set_name("L0")
    R0 = cytnx.UniTensor.zeros([D, 1, 1], device=device).set_rowrank_(0).set_name("R0")
    L0[0, 0, 0] = 1.0
    R0[D - 1, 0, 0] = 1.0
    return M, L0, R0


def _energy(A, M, L0, R0, device):
    anet = cytnx.Network()
    anet.FromString(["L: -2,-1,-3",
                      "A: -1,-4,1",
                      "M: -2,0,-4,-5",
                      "A_Conj: -3,-5,2",
                      "TOUT: 0,1,2"])
    LR = L0
    for p in range(len(A)):
        anet.PutUniTensors(["L", "A", "A_Conj", "M"],
                            [LR, A[p], A[p].Dagger().permute_(A[p].labels()), M])
        LR = anet.Launch()
    energy = cytnx.Contract(LR, R0).item()

    norm_net = cytnx.Network()
    norm_net.FromString(["L: -2,-1", "A: -1,1,-3", "A_Conj: -2,1,-4", "TOUT: -4,-3"])
    NL = cytnx.UniTensor(cytnx.ones([1, 1], device=device))
    for p in range(len(A)):
        norm_net.PutUniTensors(["L", "A", "A_Conj"],
                                [NL, A[p], A[p].Dagger().permute_(A[p].labels())])
        NL = norm_net.Launch()
    norm2 = NL.item()
    return (energy / norm2).real


def run_one(chi, L):
    device = cytnx.Device.cuda if DEVICE == "gpu" else cytnx.Device.cpu
    d = 2
    A, lbls = _build_mps(L, chi, device)
    gates = _build_gates(L, TFIM_J, TFIM_HX_FINAL, TFIM_DT, device)
    M, L0, R0 = _build_mpo(TFIM_J, TFIM_HX_FINAL, device)

    def sweep():
        for p in range(L - 1):
            psi = cytnx.Contract(A[p], A[p + 1])
            g = gates[p].clone().relabel_(["_o0", "_o1", lbls[p][1], lbls[p + 1][1]])
            psi = cytnx.Contract(psi, g)
            psi.permute_([lbls[p][0], "_o0", "_o1", lbls[p + 1][2]])
            psi.relabel_([lbls[p][0], lbls[p][1], lbls[p + 1][1], lbls[p + 1][2]])
            psi.set_rowrank_(2)
            dim_l = A[p].shape()[0]
            dim_r = A[p + 1].shape()[2]
            new_dim = min(dim_l * d, dim_r * d, chi)
            s, A[p], A[p + 1] = cytnx.linalg.Svd_truncate(psi, new_dim)
            s = s / s.Norm().item()
            A[p + 1] = cytnx.Contract(s, A[p + 1])
            A[p].set_name(f"A{p}").relabel_(lbls[p])
            A[p + 1].set_name(f"A{p+1}").relabel_(lbls[p + 1])

    for _ in range(TFIM_N_STEPS):
        sweep()
    return _energy(A, M, L0, R0, device)


@pytest.mark.timeout(GRID_POINT_TIMEOUT_SEC)
@pytest.mark.parametrize("num_sites", NUM_SITES_VALUES)
@pytest.mark.parametrize("bond_dim", BOND_DIM_VALUES)
def test_tebd_benchmark(benchmark, bond_dim, num_sites):
    energy = benchmark.pedantic(run_one, args=(bond_dim, num_sites), rounds=1, iterations=1)
    benchmark.extra_info["energy"] = energy
    assert energy == pytest.approx(REFERENCE_ENERGIES[(bond_dim, num_sites)], rel=1e-6)


@pytest.mark.cytnx_memory
@pytest.mark.limit_memory("20 MB")
def test_tebd_memory():
    energy = run_one(16, 20)
    assert energy == pytest.approx(REFERENCE_ENERGIES[(16, 20)], rel=1e-6)
