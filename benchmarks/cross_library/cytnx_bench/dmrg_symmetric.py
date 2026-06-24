"""Cytnx benchmark, algorithm class 1 (symmetric variant): block-sparse,
U(1)-total-Sz-conserving finite two-site DMRG ground-state search on the 1D
spin-1/2 Heisenberg chain.

Uses the same bond-dimension-5 finite-state-machine MPO as
`dmrg_dense.py`, here split into 5 U(1) charge sectors (start=0,
S+-pending=+2, S--pending=-2, Sz-pending=0, end=0) following the charge
bookkeeping convention of `example/DMRG/dmrg_two_sites_U1.py` (a leg with
Cytnx bond-type `BD_KET` contributes +charge and `BD_BRA` contributes
-charge to the zero-sum constraint at every nonzero MPO block). The
resulting symmetric MPO was verified against exact diagonalization for
small chains (L=4,6), matching the dense-mode MPO in `dmrg_dense.py` to
machine precision, before being used here.

CPU and GPU code paths are both written; the GPU path moves every
block-sparse MPS/MPO UniTensor to `cytnx.Device.cuda` before the sweep. It
cannot be exercised in this environment (no GPU).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cytnx

from common.metrics import (
    CSVResultWriter, StepMeasurement, StepTimeoutError, cpu_timed_block, cytnx_gpu_timed_block, time_limit,
)
from common.model import HEISENBERG_J, LANCZOS_MAXITER, N_SWEEPS, STEP_TIMEOUT_SEC, param_grid

DEVICE = "cpu"  # set to "gpu" to exercise the (untested) GPU code path below
TARGET_Q = 0  # global U(1) total-Sz sector to search within


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


def _build_mpo(J, q):
    D = 5
    bd_inner = cytnx.Bond(cytnx.BD_KET, [[0], [2], [-2], [0], [0]], [1, 1, 1, 1, 1])
    bd_phys = cytnx.Bond(cytnx.BD_KET, [[1], [-1]], [1, 1])

    M = cytnx.UniTensor([bd_inner, bd_inner.redirect(), bd_phys, bd_phys.redirect()]) \
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
    L0 = cytnx.UniTensor([bd_inner.redirect(), VbdL.redirect(), VbdL]) \
        .set_rowrank_(1).set_name("L0")
    R0 = cytnx.UniTensor([bd_inner, VbdR, VbdR.redirect()]) \
        .set_rowrank_(1).set_name("R0")
    L0.set_elem([0, 0, 0], 1)
    R0.set_elem([D - 1, 0, 0], 1)
    return M, L0, R0, bd_phys


def run_one(chi, L):
    device = cytnx.Device.cuda if DEVICE == "gpu" else cytnx.Device.cpu
    M, L0, R0, bd_phys = _build_mpo(HEISENBERG_J, TARGET_Q)
    if DEVICE == "gpu":
        M = M.to(device)
        L0 = L0.to(device)
        R0 = R0.to(device)

    A = [None for _ in range(L)]
    qcntr = 0
    cq = 1 if qcntr <= TARGET_Q else -1
    qcntr += cq

    VbdL = cytnx.Bond(cytnx.BD_KET, [[0]], [1])
    A[0] = cytnx.UniTensor([VbdL, bd_phys.redirect(), cytnx.Bond(cytnx.BD_BRA, [[qcntr]], [1])]) \
        .set_rowrank_(2).set_name("A0")
    A[0].get_block_()[0] = 1

    lbls = [["0", "1", "2"]]
    for k in range(1, L):
        B1 = A[k - 1].bonds()[2].redirect()
        B2 = A[k - 1].bonds()[1]
        cq = 1 if qcntr <= TARGET_Q else -1
        qcntr += cq
        B3 = cytnx.Bond(cytnx.BD_BRA, [[qcntr]], [1])

        A[k] = cytnx.UniTensor([B1, B2, B3]).set_rowrank_(2).set_name(f"A{k}")
        lbl = [str(2 * k), str(2 * k + 1), str(2 * k + 2)]
        A[k].relabel_(lbl)
        A[k].get_block_()[0] = 1
        lbls.append(lbl)

    if DEVICE == "gpu":
        A = [a.to(device) for a in A]

    LR = [None for _ in range(L + 1)]
    LR[0] = L0
    LR[-1] = R0

    anet = cytnx.Network()
    anet.FromString(["L: -2,-1,-3",
                      "A: -1,-4,1",
                      "M: -2,0,-4,-5",
                      "A_Conj: -3,-5,2",
                      "TOUT: 0;1,2"])
    for p in range(L - 1):
        anet.PutUniTensors(["L", "A", "A_Conj", "M"],
                            [LR[p], A[p], A[p].Dagger().permute_(A[p].labels()), M])
        LR[p + 1] = anet.Launch()
        LR[p + 1].set_name(f"LR{p+1}")

    def sweep():
        energy = None
        for p in range(L - 2, -1, -1):
            dim_l = A[p].shape()[0]
            dim_r = A[p + 1].shape()[2]
            d = A[p].shape()[1]
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
        A[0].relabel_(lbls[0])

        for p in range(L - 1):
            dim_l = A[p].shape()[0]
            dim_r = A[p + 1].shape()[2]
            d = A[p].shape()[1]
            new_dim = min(dim_l * d, dim_r * d, chi)
            psi = cytnx.Contract(A[p], A[p + 1])
            psi, energy = _optimize_psi(psi, (LR[p], M, M, LR[p + 2]), LANCZOS_MAXITER, device)
            psi.set_rowrank_(2)
            s, A[p], A[p + 1] = cytnx.linalg.Svd_truncate(psi, new_dim)
            A[p].relabel_(lbls[p])
            s = s / s.Norm().item()
            A[p + 1] = cytnx.Contract(s, A[p + 1])
            A[p + 1].relabel_(lbls[p + 1])
            A[p].set_name(f"A{p}")
            A[p + 1].set_name(f"A{p+1}")

            anet = cytnx.Network()
            anet.FromString(["L: -2,-1,-3",
                              "A: -1,-4,1",
                              "M: -2,0,-4,-5",
                              "A_Conj: -3,-5,2",
                              "TOUT: 0;1,2"])
            anet.PutUniTensors(["L", "A", "A_Conj", "M"],
                                [LR[p], A[p], A[p].Dagger().permute_(A[p].labels()), M])
            LR[p + 1] = anet.Launch()
            LR[p + 1].set_name(f"LR{p+1}")

        A[-1].set_rowrank_(2)
        _, A[-1] = cytnx.linalg.Gesvd(A[-1], is_U=True, is_vT=False)
        A[-1].set_name(f"A{L-1}").relabel_(lbls[-1])
        return energy

    timed_block = cytnx_gpu_timed_block if DEVICE == "gpu" else cpu_timed_block
    energy = None
    with timed_block() as r:
        for _ in range(N_SWEEPS):
            energy = sweep()
    step_time = r["time_sec"] / N_SWEEPS
    return step_time, r["peak_mem_mb"], energy


def main(out_csv):
    writer = CSVResultWriter(out_csv)
    for chi, L in param_grid():
        try:
            with time_limit(STEP_TIMEOUT_SEC):
                step_time, peak_mem_mb, energy = run_one(chi, L)
        except StepTimeoutError:
            print(f"[cytnx/dmrg_symmetric] chi={chi} L={L} skipped (exceeded {STEP_TIMEOUT_SEC}s)")
            continue
        writer.write(StepMeasurement(
            library="cytnx", algorithm="dmrg_symmetric", symmetry="u1",
            device=DEVICE, backend="cytnx", L=L, chi=chi,
            step_time_sec=step_time, peak_mem_mb=peak_mem_mb, answer=energy,
        ))
        print(f"[cytnx/dmrg_symmetric] chi={chi} L={L} "
              f"time/sweep={step_time:.4f}s peak_mem={peak_mem_mb:.1f}MB energy={energy:.6f}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results/cytnx_dmrg_symmetric.csv"
    main(out)
