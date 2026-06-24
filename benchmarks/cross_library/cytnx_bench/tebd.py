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
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cytnx

from common.metrics import (
    CSVResultWriter, StepMeasurement, StepTimeoutError, completed_keys, cpu_timed_block, cytnx_gpu_timed_block,
    time_limit,
)
from common.model import STEP_TIMEOUT_SEC, TFIM_DT, TFIM_HX_FINAL, TFIM_J, TFIM_N_STEPS, param_grid

DEVICE = "cpu"  # set to "gpu" to exercise the (untested) GPU code path below


def _build_gate(J, hx, dt, w_left, w_right, device):
    Sz = cytnx.physics.pauli("z").real()
    Sx = cytnx.physics.pauli("x").real()
    I = cytnx.eye(2)
    TFterm = w_left * cytnx.linalg.Kron(Sx, I) + w_right * cytnx.linalg.Kron(I, Sx)
    ZZterm = cytnx.linalg.Kron(Sz, Sz)
    H = -hx * TFterm - J * ZZterm
    eH = cytnx.linalg.ExpH(H, -1j * dt)
    eH.reshape_(2, 2, 2, 2)
    gate = cytnx.UniTensor(eH)
    if device == "gpu":
        gate = gate.to(cytnx.Device.cuda)
    return gate


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
        A[k] = cytnx.UniTensor.zeros([1, d, 1]).set_rowrank_(2).relabel_(lbls[k]).set_name(f"A{k}")
        A[k].set_elem([0, 0, 0], 1.0)
        if device == "gpu":
            A[k] = A[k].to(cytnx.Device.cuda)
    return A, lbls


def _build_mpo(J, hx):
    D = 3
    Sz = cytnx.physics.pauli("z").real()
    Sx = cytnx.physics.pauli("x").real()
    eye = cytnx.eye(2)

    M = cytnx.zeros([D, D, 2, 2])
    M[0, 0] = eye
    M[0, 1] = Sz
    M[0, 2] = -hx * Sx
    M[1, 2] = -J * Sz
    M[D - 1, D - 1] = eye
    M = cytnx.UniTensor(M, 0).set_name("MPO")

    L0 = cytnx.UniTensor.zeros([D, 1, 1]).set_rowrank_(0).set_name("L0")
    R0 = cytnx.UniTensor.zeros([D, 1, 1]).set_rowrank_(0).set_name("R0")
    L0[0, 0, 0] = 1.0
    R0[D - 1, 0, 0] = 1.0
    return M, L0, R0


def _energy(A, M, L0, R0, device):
    if device == "gpu":
        M = M.to(cytnx.Device.cuda)
        L0 = L0.to(cytnx.Device.cuda)
        R0 = R0.to(cytnx.Device.cuda)

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
    NL = cytnx.UniTensor(cytnx.ones([1, 1]))
    for p in range(len(A)):
        norm_net.PutUniTensors(["L", "A", "A_Conj"],
                                [NL, A[p], A[p].Dagger().permute_(A[p].labels())])
        NL = norm_net.Launch()
    norm2 = NL.item()
    return (energy / norm2).real


def run_one(chi, L):
    timed_block = cytnx_gpu_timed_block if DEVICE == "gpu" else cpu_timed_block
    with timed_block() as r:
        device = "gpu" if DEVICE == "gpu" else "cpu"
        d = 2
        A, lbls = _build_mps(L, chi, device)
        gates = _build_gates(L, TFIM_J, TFIM_HX_FINAL, TFIM_DT, device)
        M, L0, R0 = _build_mpo(TFIM_J, TFIM_HX_FINAL)

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

        t0 = time.perf_counter()
        for _ in range(TFIM_N_STEPS):
            sweep()
        loop_time = time.perf_counter() - t0
        energy = _energy(A, M, L0, R0, device)
    step_time = loop_time / TFIM_N_STEPS
    return step_time, r["peak_mem_mb"], energy


def main(out_csv):
    writer = CSVResultWriter(out_csv)
    done = completed_keys(out_csv, "chi", "L")
    for chi, L in param_grid():
        if (str(chi), str(L)) in done:
            continue
        try:
            with time_limit(STEP_TIMEOUT_SEC):
                step_time, peak_mem_mb, energy = run_one(chi, L)
        except StepTimeoutError:
            print(f"[cytnx/tebd_quench] chi={chi} L={L} skipped (exceeded {STEP_TIMEOUT_SEC}s)")
            continue
        writer.write(StepMeasurement(
            library="cytnx", algorithm="tebd_quench", symmetry="dense",
            device=DEVICE, backend="cytnx", L=L, chi=chi,
            step_time_sec=step_time, peak_mem_mb=peak_mem_mb, answer=energy,
        ))
        print(f"[cytnx/tebd_quench] chi={chi} L={L} "
              f"time/step={step_time:.4f}s peak_mem={peak_mem_mb:.1f}MB energy={energy:.6f}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results/cytnx_tebd.csv"
    main(out)
