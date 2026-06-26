"""Cytnx benchmark, algorithm class 1: finite two-site DMRG, dense mode (no
conserved quantum numbers) on the 1D spin-1/2 Heisenberg chain.

Implements the standard bond-dimension-5 finite-state-machine MPO for the
isotropic Heisenberg coupling J*S_i.S_j = (J/2)*(S+_i S-_j + S-_i S+_j) +
J*Sz_i Sz_j, and a textbook two-site DMRG sweep (local eigensolve via
Cytnx's `LinOp` + `Lanczos`, truncation via `Svd_truncate`) adapted from
`example/DMRG/dmrg_two_sites_dense.py`. The MPO was verified against exact
diagonalization for small chains (L=4,6) before being used here.

CPU and GPU code paths are both written; the GPU path moves every MPS/MPO
UniTensor to `cytnx.Device.cuda` before the sweep. It cannot be exercised in
this environment (no GPU).

Run timing with `pytest --benchmark-only test_dmrg_dense.py`, memory with
`pytest --memray test_dmrg_dense.py`.
"""
import pytest

import cytnx

from common.model import BOND_DIM_VALUES, HEISENBERG_J, LANCZOS_MAXITER, NUM_SITES_VALUES, N_SWEEPS, GRID_POINT_TIMEOUT_SEC

DEVICE = "cpu"  # set to "gpu" to exercise the (untested) GPU code path below

REFERENCE_ENERGIES = {
    (16, 20): -8.682468366628518,
    (16, 30): -13.111312403537152,
    (16, 50): -21.971805310867897,
    (32, 20): -8.68247331965388,
    (32, 30): -13.111355520184109,
    (32, 50): -21.972106487235166,
    (64, 20): -8.682473334397889,
    (64, 30): -13.111355758487786,
    (64, 50): -21.97211027157147,
}


class _Hxx(cytnx.LinOp):
    def __init__(self, anet, psidim, device):
        cytnx.LinOp.__init__(self, "mv", psidim, cytnx.Type.Double, device)
        self.anet = anet

    def matvec(self, v):
        lbl = v.labels()
        self.anet.PutUniTensor("psi", v)
        out = self.anet.Launch()
        out.relabel_(lbl)
        return out


def _optimize_psi(psi, functArgs, maxit, device):
    L, M1, M2, R = functArgs
    anet = cytnx.Network()
    anet.FromString(["psi: -1,-2,-3,-4",
                      "L: -5,-1,0",
                      "R: -7,-4,3",
                      "M1: -5,-6,-2,1",
                      "M2: -6,-7,-3,2",
                      "TOUT: 0,1;2,3"])
    anet.PutUniTensors(["L", "M1", "M2", "R"], [L, M1, M2, R])
    H = _Hxx(anet, psi.shape()[0] * psi.shape()[1] * psi.shape()[2] * psi.shape()[3], device)
    energy, psivec = cytnx.linalg.Lanczos(Hop=H, method="Gnd", Maxiter=maxit, CvgCrit=1e-12, Tin=psi)
    return psivec, energy[0].item()


def _build_mpo(J):
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
    return M, L0, R0


def run_one(chi, L):
    device = cytnx.Device.cuda if DEVICE == "gpu" else cytnx.Device.cpu
    d = 2
    M, L0, R0 = _build_mpo(HEISENBERG_J)
    if DEVICE == "gpu":
        M = M.to(device)
        L0 = L0.to(device)
        R0 = R0.to(device)

    A = [None for _ in range(L)]
    A[0] = cytnx.UniTensor.normal([1, d, min(chi, d)], 0., 1.).set_rowrank_(2)
    A[0].relabel_(["0", "1", "2"]).set_name("A0")
    if DEVICE == "gpu":
        A[0] = A[0].to(device)

    lbls = [["0", "1", "2"]]
    for k in range(1, L):
        dim1 = A[k - 1].shape()[2]
        dim2 = d
        dim3 = min(min(chi, A[k - 1].shape()[2] * d), d ** (L - k - 1))
        A[k] = cytnx.UniTensor.normal([dim1, dim2, dim3], 0., 1.).set_rowrank_(2).set_name(f"A{k}")
        lbl = [str(2 * k), str(2 * k + 1), str(2 * k + 2)]
        A[k].relabel_(lbl)
        if DEVICE == "gpu":
            A[k] = A[k].to(device)
        lbls.append(lbl)

    LR = [None for _ in range(L + 1)]
    LR[0] = L0
    LR[-1] = R0

    for p in range(L - 1):
        s, A[p], vt = cytnx.linalg.Gesvd(A[p])
        A[p + 1] = cytnx.Contract(cytnx.Contract(s, vt), A[p + 1])
        A[p].set_name(f"A{p}")
        A[p + 1].set_name(f"A{p+1}")

        anet = cytnx.Network()
        anet.FromString(["L: -2,-1,-3",
                          "A: -1,-4,1",
                          "M: -2,0,-4,-5",
                          "A_Conj: -3,-5,2",
                          "TOUT: 0,1,2"])
        anet.PutUniTensors(["L", "A", "A_Conj", "M"],
                            [LR[p], A[p], A[p].Dagger().permute_(A[p].labels()), M])
        LR[p + 1] = anet.Launch()
        LR[p + 1].set_name(f"LR{p+1}")
        A[p].relabel_(lbls[p])
        A[p + 1].relabel_(lbls[p + 1])

    _, A[-1] = cytnx.linalg.Gesvd(A[-1], is_U=True, is_vT=False)
    A[-1].set_name(f"A{L-1}").relabel_(lbls[-1])

    def sweep():
        energy = None
        for p in range(L - 2, -1, -1):
            dim_l = A[p].shape()[0]
            dim_r = A[p + 1].shape()[2]
            new_dim = min(dim_l * d, dim_r * d, chi)
            psi = cytnx.Contract(A[p], A[p + 1])
            psi, energy = _optimize_psi(psi, (LR[p], M, M, LR[p + 2]), LANCZOS_MAXITER, device)
            psi.set_rowrank_(2)
            s, A[p], A[p + 1] = cytnx.linalg.Svd_truncate(psi, new_dim)
            A[p + 1].set_name(f"A{p+1}").relabel_(lbls[p + 1])
            s = s / s.Norm().item()
            A[p] = cytnx.Contract(A[p], s)
            A[p].set_name(f"A{p}").relabel_(lbls[p])

            anet = cytnx.Network()
            anet.FromString(["R: -2,-1,-3",
                              "B: 1,-4,-1",
                              "M: 0,-2,-4,-5",
                              "B_Conj: 2,-5,-3",
                              "TOUT: 0;1,2"])
            anet.PutUniTensors(["R", "B", "M", "B_Conj"],
                                [LR[p + 2], A[p + 1], M, A[p + 1].Dagger().permute_(A[p + 1].labels())])
            LR[p + 1] = anet.Launch()
            LR[p + 1].set_name(f"LR{p+1}")

        A[0].set_rowrank_(1)
        _, A[0] = cytnx.linalg.Gesvd(A[0], is_U=False, is_vT=True)
        A[0].set_name("A0").relabel_(lbls[0])

        for p in range(L - 1):
            dim_l = A[p].shape()[0]
            dim_r = A[p + 1].shape()[2]
            new_dim = min(dim_l * d, dim_r * d, chi)
            psi = cytnx.Contract(A[p], A[p + 1])
            psi, energy = _optimize_psi(psi, (LR[p], M, M, LR[p + 2]), LANCZOS_MAXITER, device)
            psi.set_rowrank_(2)
            s, A[p], A[p + 1] = cytnx.linalg.Svd_truncate(psi, new_dim)
            A[p].set_name(f"A{p}").relabel_(lbls[p])
            s = s / s.Norm().item()
            A[p + 1] = cytnx.Contract(s, A[p + 1])
            A[p + 1].set_name(f"A{p+1}").relabel_(lbls[p + 1])

            anet = cytnx.Network()
            anet.FromString(["L: -2,-1,-3",
                              "A: -1,-4,1",
                              "M: -2,0,-4,-5",
                              "A_Conj: -3,-5,2",
                              "TOUT: 0,1,2"])
            anet.PutUniTensors(["L", "A", "A_Conj", "M"],
                                [LR[p], A[p], A[p].Dagger().permute_(A[p].labels()), M])
            LR[p + 1] = anet.Launch()
            LR[p + 1].set_name(f"LR{p+1}")

        A[-1].set_rowrank_(2)
        _, A[-1] = cytnx.linalg.Gesvd(A[-1], is_U=True, is_vT=False)
        A[-1].set_name(f"A{L-1}").relabel_(lbls[-1])
        return energy

    energy = None
    for _ in range(N_SWEEPS):
        energy = sweep()
    return energy


@pytest.mark.timeout(GRID_POINT_TIMEOUT_SEC)
@pytest.mark.parametrize("num_sites", NUM_SITES_VALUES)
@pytest.mark.parametrize("bond_dim", BOND_DIM_VALUES)
def test_dmrg_dense_benchmark(benchmark, bond_dim, num_sites):
    energy = benchmark.pedantic(run_one, args=(bond_dim, num_sites), rounds=1, iterations=1)
    benchmark.extra_info["energy"] = energy
    assert energy == pytest.approx(REFERENCE_ENERGIES[(bond_dim, num_sites)], rel=1e-6)


@pytest.mark.cytnx_memory
@pytest.mark.parametrize("num_sites", NUM_SITES_VALUES)
@pytest.mark.parametrize("bond_dim", BOND_DIM_VALUES)
def test_dmrg_dense_memory(bond_dim, num_sites):
    energy = run_one(bond_dim, num_sites)
    assert energy == pytest.approx(REFERENCE_ENERGIES[(bond_dim, num_sites)], rel=1e-6)
