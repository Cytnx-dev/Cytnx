"""Cytnx benchmark, algorithm class 2: real-time evolution of a 1D
transverse-field Ising chain after a field quench, using a hand-rolled TEBD
sweep built from Cytnx's own `UniTensor`/`Contract`/`Svd_truncate` API (Cytnx
has no built-in TEBD engine, unlike TeNPy/quimb).

Each Trotter step is the symmetric (Strang) product exp(-i*dt/2*h_{L-2})...
exp(-i*dt/2*h_0) * exp(-i*dt/2*h_0)...exp(-i*dt/2*h_{L-2}) -- a forward
left-to-right half-step sweep over every bond followed by a backward
right-to-left half-step sweep over every bond, each absorbing the post-SVD
singular-value tensor into the not-yet-visited neighbor (right on the
forward pass, left on the backward pass) so the orthogonality center moves
with the sweep direction. This palindromic forward/backward composition is
second-order accurate in dt for an arbitrary ordered decomposition of H
into bond terms h_p, matching the order=2 even/odd Suzuki-Trotter splitting
used by `tenpy_bench/test_tebd.py` and `quimb_bench/test_tebd.py` in
accuracy order, though not in the specific grouping of bond terms. The
on-site field term -hx*Sx is split between the two bond gates that touch
each site (half-weight on interior bonds, full weight on the two
chain-boundary bonds) so that summing the per-bond gates' field
contributions over a full sweep reproduces -hx*sum(Sx_i) exactly once per
site rather than twice for interior sites. The initial state is the
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

from common.model import BOND_DIM_VALUES, NUM_SITES_VALUES, GRID_POINT_TIMEOUT_SEC, SVD_CUTOFF, TFIM_DT, TFIM_HX_FINAL, TFIM_J, TFIM_N_STEPS

DEVICE = "cpu"  # set to "gpu" to exercise the (untested) GPU code path below

REFERENCE_ENERGIES = {
    (16, 20): -19.0002116253495,
    (16, 30): -29.000296405051223,
    (16, 50): -49.00046596445489,
    (32, 20): -19.000211625349483,
    (32, 30): -29.000296405051227,
    (32, 50): -49.00046596445477,
    (64, 20): -19.000211625349483,
    (64, 30): -29.000296405051227,
    (64, 50): -49.00046596445477,
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
    # Every interior bond (0 < p < L-2) shares the same (0.5, 0.5) on-site-field
    # split, so only the two boundary bonds need a distinct exp(-i*dt*h_bond);
    # the single interior gate is reused across all interior bonds.
    if L == 2:
        return [_build_gate(J, hx, dt, 1.0, 1.0, device)]
    left_gate = _build_gate(J, hx, dt, 1.0, 0.5, device)
    right_gate = _build_gate(J, hx, dt, 0.5, 1.0, device)
    interior_gate = _build_gate(J, hx, dt, 0.5, 0.5, device) if L > 3 else None
    return [left_gate] + [interior_gate] * (L - 3) + [right_gate]


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
    base_gates = _build_gates(L, TFIM_J, TFIM_HX_FINAL, TFIM_DT / 2, device)
    # Bond labels are fixed by _build_mps and restored after every truncation
    # below, so each gate only ever needs to be relabeled once, not on every
    # Trotter step.
    gates = [base_gates[p].clone().relabel_(["_o0", "_o1", lbls[p][1], lbls[p + 1][1]])
             for p in range(L - 1)]
    M, L0, R0 = _build_mpo(TFIM_J, TFIM_HX_FINAL, device)

    def apply_gate(p):
        psi = cytnx.Contract(A[p], A[p + 1])
        psi = cytnx.Contract(psi, gates[p])
        psi.permute_([lbls[p][0], "_o0", "_o1", lbls[p + 1][2]])
        psi.relabel_([lbls[p][0], lbls[p][1], lbls[p + 1][1], lbls[p + 1][2]])
        psi.set_rowrank_(2)
        dim_l = A[p].shape()[0]
        dim_r = A[p + 1].shape()[2]
        new_dim = min(dim_l * d, dim_r * d, chi)
        s, A[p], A[p + 1] = cytnx.linalg.Svd_truncate(psi, new_dim, err=SVD_CUTOFF)
        return s

    def sweep():
        for p in range(L - 1):
            s = apply_gate(p)
            s = s / s.Norm().item()
            A[p + 1] = cytnx.Contract(s, A[p + 1])
            A[p].set_name(f"A{p}").relabel_(lbls[p])
            A[p + 1].set_name(f"A{p+1}").relabel_(lbls[p + 1])
        for p in range(L - 2, -1, -1):
            s = apply_gate(p)
            s = s / s.Norm().item()
            A[p] = cytnx.Contract(A[p], s)
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
@pytest.mark.parametrize("num_sites", NUM_SITES_VALUES)
@pytest.mark.parametrize("bond_dim", BOND_DIM_VALUES)
def test_tebd_memory(bond_dim, num_sites):
    energy = run_one(bond_dim, num_sites)
    assert energy == pytest.approx(REFERENCE_ENERGIES[(bond_dim, num_sites)], rel=1e-6)
