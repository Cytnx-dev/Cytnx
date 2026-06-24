"""quimb benchmark, algorithm class 2: real-time evolution of a 1D
transverse-field Ising chain after a field quench, using quimb's built-in
`TEBD` engine.

CPU and GPU code paths are both written; the GPU path moves the initial
MPS to the device array backend before stepping. It cannot be exercised
in this environment (no GPU).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import quimb.tensor as qtn

from common.metrics import (
    CSVResultWriter, StepMeasurement, StepTimeoutError, completed_keys, cpu_timed_block, time_limit,
    torch_gpu_timed_block,
)
from common.model import STEP_TIMEOUT_SEC, TFIM_DT, TFIM_HX_FINAL, TFIM_J, TFIM_N_STEPS, param_grid

DEVICE = "cpu"  # set to "gpu" to exercise the (untested) GPU code path below


def build(chi, L):
    H = qtn.ham_1d_ising(L, j=TFIM_J, bx=TFIM_HX_FINAL, cyclic=False)
    psi0 = qtn.MPS_computational_state("0" * L)
    if DEVICE == "gpu":
        import torch
        psi0.apply_to_arrays(lambda x: torch.as_tensor(x, device="cuda"))
    tebd = qtn.TEBD(psi0, H, dt=TFIM_DT, progbar=False)
    tebd.split_opts["cutoff"] = 1e-10
    tebd.split_opts["max_bond"] = chi
    return tebd


def run_one(chi, L):
    tebd = build(chi, L)
    timed_block = torch_gpu_timed_block if DEVICE == "gpu" else cpu_timed_block
    with timed_block() as r:
        for _ in range(TFIM_N_STEPS):
            tebd.step(order=2, dt=TFIM_DT)
    step_time = r["time_sec"] / TFIM_N_STEPS
    H_mpo = qtn.MPO_ham_ising(L, j=TFIM_J, bx=TFIM_HX_FINAL, cyclic=False)
    energy = tebd.pt.H @ (H_mpo.apply(tebd.pt))
    return step_time, r["peak_mem_mb"], float(energy.real)


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
            print(f"[quimb/tebd_quench] chi={chi} L={L} skipped (exceeded {STEP_TIMEOUT_SEC}s)")
            continue
        writer.write(StepMeasurement(
            library="quimb", algorithm="tebd_quench", symmetry="dense",
            device=DEVICE, backend="numpy" if DEVICE == "cpu" else "torch",
            L=L, chi=chi, step_time_sec=step_time, peak_mem_mb=peak_mem_mb, answer=energy,
        ))
        print(f"[quimb/tebd_quench] chi={chi} L={L} "
              f"time/step={step_time:.4f}s peak_mem={peak_mem_mb:.1f}MB energy={energy:.6f}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results/quimb_tebd.csv"
    main(out)
