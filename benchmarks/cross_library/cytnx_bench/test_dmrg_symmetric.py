"""Cytnx benchmark, algorithm class 1 (symmetric variant): block-sparse,
U(1)-total-Sz-conserving finite two-site DMRG ground-state search on the 1D
spin-1/2 Heisenberg chain.

Uses the same bond-dimension-5 finite-state-machine MPO as
`test_dmrg_dense.py`, here split into 5 U(1) charge sectors (start=0,
S+-pending=+2, S--pending=-2, Sz-pending=0, end=0) following the charge
bookkeeping convention of `example/DMRG/dmrg_two_sites_U1.py` (a leg with
Cytnx bond-type `BD_KET` contributes +charge and `BD_BRA` contributes
-charge to the zero-sum constraint at every nonzero MPO block). The
resulting symmetric MPO was verified against exact diagonalization for
small chains (L=4,6), matching the dense-mode MPO in `test_dmrg_dense.py`
to machine precision, before being used here.

CPU and GPU code paths are both written; the GPU path moves every
block-sparse MPS/MPO UniTensor to `cytnx.Device.cuda` before the sweep. It
cannot be exercised in this environment (no GPU).

Run timing with `pytest --benchmark-only test_dmrg_symmetric.py`, memory
with `pytest --memray test_dmrg_symmetric.py`.
"""
import pytest

import cytnx

from common.model import BOND_DIM_VALUES, HEISENBERG_J, LANCZOS_MAXITER, NUM_SITES_VALUES, N_SWEEPS, GRID_POINT_TIMEOUT_SEC, SVD_CUTOFF

DEVICE = "cpu"  # set to "gpu" to exercise the (untested) GPU code path below
TARGET_Q = 0  # global U(1) total-Sz sector to search within

REFERENCE_ENERGIES = {
    (16, 20): -8.682468456356823,
    (16, 30): -13.111313454922696,
    (16, 50): -21.971813615699435,
    (32, 20): -8.682473319689738,
    (32, 30): -13.111355524202278,
    (32, 50): -21.972106512010665,
    (64, 20): -8.682473334397892,
    (64, 30): -13.11135575848872,
    (64, 50): -21.972110271890013,
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


def _stored_numel(ut):
    # BlockUniTensor.shape() returns each bond's nominal (sum-of-sectors) extent,
    # not the element count actually stored in the nonzero charge blocks, so the
    # active dimension of a block-sparse psi must be summed from its own blocks.
    total = 0
    for block in ut.get_blocks_():
        n = 1
        for d in block.shape():
            n *= d
        total += n
    return total


def _h_eff_network():
    anet = cytnx.Network()
    anet.FromString(["psi: -1,-2,-3,-4",
                      "L: -5,-1,0",
                      "R: -7,-4,3",
                      "M1: -5,-6,-2,1",
                      "M2: -6,-7,-3,2",
                      "TOUT: 0,1;2,3"])
    return anet


def _l_update_network():
    anet = cytnx.Network()
    anet.FromString(["L: -2,-1,-3",
                      "A: -1,-4,1",
                      "M: -2,0,-4,-5",
                      "A_Conj: -3,-5,2",
                      "TOUT: 0;1,2"])
    return anet


def _r_update_network():
    anet = cytnx.Network()
    anet.FromString(["R: -2,-1,-3",
                      "B: 1,-4,-1",
                      "M: 0,-2,-4,-5",
                      "B_Conj: 2,-5,-3",
                      "TOUT: 0;1,2"])
    return anet


def _optimize_psi(anet, psi, L, M1, M2, R, maxit, device):
    anet.PutUniTensors(["L", "M1", "M2", "R"], [L, M1, M2, R])
    H = _Hxx(anet, _stored_numel(psi), device)
    energy, psivec = cytnx.linalg.Lanczos(Hop=H, method="Gnd", Maxiter=maxit, CvgCrit=1e-12, Tin=psi)
    return psivec, energy[0].item()


def _build_mpo(J, q, device):
    D = 5
    bd_inner = cytnx.Bond(cytnx.BD_KET, [[0], [2], [-2], [0], [0]], [1, 1, 1, 1, 1])
    bd_phys = cytnx.Bond(cytnx.BD_KET, [[1], [-1]], [1, 1])

    M = cytnx.UniTensor([bd_inner, bd_inner.redirect(), bd_phys, bd_phys.redirect()], device=device) \
        .set_rowrank_(2).set_name("MPO")

    M.set_elem([0, 0, 0, 0], 1)
    M.set_elem([0, 0, 1, 1], 1)
    M.set_elem([D - 1, D - 1, 0, 0], 1)
    M.set_elem([D - 1, D - 1, 1, 1], 1)

    M.set_elem([0, 1, 0, 1], 1)              # S+
    M.set_elem([0, 2, 1, 0], 1)              # S-
    M.set_elem([0, 3, 0, 0], 0.5)            # Sz
    M.set_elem([0, 3, 1, 1], -0.5)           # Sz

    M.set_elem([1, D - 1, 1, 0], J / 2.0)    # (J/2) S-
    M.set_elem([2, D - 1, 0, 1], J / 2.0)    # (J/2) S+
    M.set_elem([3, D - 1, 0, 0], J * 0.5)    # J Sz
    M.set_elem([3, D - 1, 1, 1], -J * 0.5)   # J Sz

    VbdL = cytnx.Bond(cytnx.BD_KET, [[0]], [1])
    VbdR = cytnx.Bond(cytnx.BD_KET, [[q]], [1])
    L0 = cytnx.UniTensor([bd_inner.redirect(), VbdL.redirect(), VbdL], device=device) \
        .set_rowrank_(1).set_name("L0")
    R0 = cytnx.UniTensor([bd_inner, VbdR, VbdR.redirect()], device=device) \
        .set_rowrank_(1).set_name("R0")
    L0.set_elem([0, 0, 0], 1)
    R0.set_elem([D - 1, 0, 0], 1)
    return M, L0, R0, bd_phys


def run_one(chi, L):
    device = cytnx.Device.cuda if DEVICE == "gpu" else cytnx.Device.cpu
    M, L0, R0, bd_phys = _build_mpo(HEISENBERG_J, TARGET_Q, device)

    A = [None for _ in range(L)]
    running_charge = 0
    charge_step = 1 if running_charge <= TARGET_Q else -1
    running_charge += charge_step

    VbdL = cytnx.Bond(cytnx.BD_KET, [[0]], [1])
    A[0] = cytnx.UniTensor([VbdL, bd_phys.redirect(), cytnx.Bond(cytnx.BD_BRA, [[running_charge]], [1])], device=device) \
        .set_rowrank_(2).set_name("A0")
    A[0].get_block_()[0] = 1

    lbls = [["0", "1", "2"]]
    for k in range(1, L):
        B1 = A[k - 1].bonds()[2].redirect()
        B2 = A[k - 1].bonds()[1]
        charge_step = 1 if running_charge <= TARGET_Q else -1
        running_charge += charge_step
        B3 = cytnx.Bond(cytnx.BD_BRA, [[running_charge]], [1])

        A[k] = cytnx.UniTensor([B1, B2, B3], device=device).set_rowrank_(2).set_name(f"A{k}")
        lbl = [str(2 * k), str(2 * k + 1), str(2 * k + 2)]
        A[k].relabel_(lbl)
        A[k].get_block_()[0] = 1
        lbls.append(lbl)

    LR = [None for _ in range(L + 1)]
    LR[0] = L0
    LR[-1] = R0

    h_eff_net = _h_eff_network()
    l_update_net = _l_update_network()
    r_update_net = _r_update_network()

    for p in range(L - 1):
        l_update_net.PutUniTensors(["L", "A", "A_Conj", "M"],
                                    [LR[p], A[p], A[p].Dagger().permute_(A[p].labels()), M])
        LR[p + 1] = l_update_net.Launch()
        LR[p + 1].set_name(f"LR{p+1}")

    def sweep():
        energy = None
        for p in range(L - 2, -1, -1):
            dim_l = A[p].shape()[0]
            dim_r = A[p + 1].shape()[2]
            d = A[p].shape()[1]
            new_dim = min(dim_l * d, dim_r * d, chi)
            psi = cytnx.Contract(A[p], A[p + 1])
            psi, energy = _optimize_psi(h_eff_net, psi, LR[p], M, M, LR[p + 2], LANCZOS_MAXITER, device)
            psi.set_rowrank_(2)
            s, A[p], A[p + 1] = cytnx.linalg.Svd_truncate(psi, new_dim, err=SVD_CUTOFF)
            A[p + 1].set_name(f"A{p+1}").relabel_(lbls[p + 1])
            s = s / s.Norm().item()
            A[p] = cytnx.Contract(A[p], s)
            A[p].set_name(f"A{p}").relabel_(lbls[p])

            r_update_net.PutUniTensors(["R", "B", "M", "B_Conj"],
                                        [LR[p + 2], A[p + 1], M, A[p + 1].Dagger().permute_(A[p + 1].labels())])
            LR[p + 1] = r_update_net.Launch()
            LR[p + 1].set_name(f"LR{p+1}")

        A[0].set_rowrank_(1)
        _, A[0] = cytnx.linalg.Gesvd(A[0], is_U=False, is_vT=True)
        A[0].relabel_(lbls[0])

        for p in range(L - 1):
            dim_l = A[p].shape()[0]
            dim_r = A[p + 1].shape()[2]
            d = A[p].shape()[1]
            new_dim = min(dim_l * d, dim_r * d, chi)
            psi = cytnx.Contract(A[p], A[p + 1])
            psi, energy = _optimize_psi(h_eff_net, psi, LR[p], M, M, LR[p + 2], LANCZOS_MAXITER, device)
            psi.set_rowrank_(2)
            s, A[p], A[p + 1] = cytnx.linalg.Svd_truncate(psi, new_dim, err=SVD_CUTOFF)
            A[p].relabel_(lbls[p])
            s = s / s.Norm().item()
            A[p + 1] = cytnx.Contract(s, A[p + 1])
            A[p + 1].relabel_(lbls[p + 1])
            A[p].set_name(f"A{p}")
            A[p + 1].set_name(f"A{p+1}")

            l_update_net.PutUniTensors(["L", "A", "A_Conj", "M"],
                                        [LR[p], A[p], A[p].Dagger().permute_(A[p].labels()), M])
            LR[p + 1] = l_update_net.Launch()
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
def test_dmrg_symmetric_benchmark(benchmark, bond_dim, num_sites):
    energy = benchmark.pedantic(run_one, args=(bond_dim, num_sites), rounds=1, iterations=1)
    benchmark.extra_info["energy"] = energy
    assert energy == pytest.approx(REFERENCE_ENERGIES[(bond_dim, num_sites)], rel=1e-4)


@pytest.mark.cytnx_memory
@pytest.mark.limit_memory("20 MB")
@pytest.mark.parametrize("num_sites", NUM_SITES_VALUES)
@pytest.mark.parametrize("bond_dim", BOND_DIM_VALUES)
def test_dmrg_symmetric_memory(bond_dim, num_sites):
    energy = run_one(bond_dim, num_sites)
    assert energy == pytest.approx(REFERENCE_ENERGIES[(bond_dim, num_sites)], rel=1e-4)
