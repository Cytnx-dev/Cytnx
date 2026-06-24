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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cytnx

from common.metrics import CSVResultWriter, StepMeasurement, cpu_timed_block, cytnx_gpu_timed_block
from common.model import TFIM_DT, TFIM_HX_FINAL, TFIM_J, TFIM_N_STEPS, param_grid

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


def run_one(chi, L):
    device = "gpu" if DEVICE == "gpu" else "cpu"
    d = 2
    A, lbls = _build_mps(L, chi, device)
    gates = _build_gates(L, TFIM_J, TFIM_HX_FINAL, TFIM_DT, device)

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

    timed_block = cytnx_gpu_timed_block if DEVICE == "gpu" else cpu_timed_block
    with timed_block() as r:
        for _ in range(TFIM_N_STEPS):
            sweep()
    step_time = r["time_sec"] / TFIM_N_STEPS
    return step_time, r["peak_mem_mb"]


def main(out_csv):
    writer = CSVResultWriter(out_csv)
    for chi, L in param_grid():
        step_time, peak_mem_mb = run_one(chi, L)
        writer.write(StepMeasurement(
            library="cytnx", algorithm="tebd_quench", symmetry="dense",
            device=DEVICE, backend="cytnx", L=L, chi=chi,
            step_time_sec=step_time, peak_mem_mb=peak_mem_mb,
        ))
        print(f"[cytnx/tebd_quench] chi={chi} L={L} "
              f"time/step={step_time:.4f}s peak_mem={peak_mem_mb:.1f}MB")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results/cytnx_tebd.csv"
    main(out)
