"""quimb benchmark, algorithm class 1: finite two-site DMRG, dense mode
(no conserved quantum numbers) on the 1D spin-1/2 Heisenberg chain, using
quimb's built-in `DMRG2` engine.

CPU and GPU code paths are both written; the GPU path moves every MPO/MPS
tensor to the device array backend before the sweep (mirrors quimb's
documented `to_backend`/`apply_to_arrays` pattern with cupy or torch). It
cannot be exercised in this environment (no GPU).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import quimb.tensor as qtn

from common.metrics import CSVResultWriter, StepMeasurement, cpu_timed_block, torch_gpu_timed_block
from common.model import HEISENBERG_J, N_SWEEPS, param_grid

DEVICE = "cpu"  # set to "gpu" to exercise the (untested) GPU code path below


def build(chi, L):
    H = qtn.MPO_ham_heis(L, j=HEISENBERG_J, cyclic=False)
    if DEVICE == "gpu":
        import torch
        H.apply_to_arrays(lambda x: torch.as_tensor(x, device="cuda"))
    dmrg = qtn.DMRG2(H, bond_dims=[chi], cutoffs=1e-10)
    return dmrg


def run_one(chi, L):
    dmrg = build(chi, L)
    timed_block = torch_gpu_timed_block if DEVICE == "gpu" else cpu_timed_block
    with timed_block() as r:
        dmrg.solve(tol=1e-6, max_sweeps=N_SWEEPS, verbosity=0)
    step_time = r["time_sec"] / N_SWEEPS
    return step_time, r["peak_mem_mb"]


def main(out_csv):
    writer = CSVResultWriter(out_csv)
    for chi, L in param_grid():
        step_time, peak_mem_mb = run_one(chi, L)
        writer.write(StepMeasurement(
            library="quimb", algorithm="dmrg_dense", symmetry="dense",
            device=DEVICE, backend="numpy" if DEVICE == "cpu" else "torch",
            L=L, chi=chi, step_time_sec=step_time, peak_mem_mb=peak_mem_mb,
        ))
        print(f"[quimb/dmrg_dense] chi={chi} L={L} "
              f"time/sweep={step_time:.4f}s peak_mem={peak_mem_mb:.1f}MB")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results/quimb_dmrg_dense.csv"
    main(out)
