"""TeNPy benchmark, algorithm class 1: finite two-site DMRG, dense mode
(no conserved quantum numbers) on the 1D spin-1/2 Heisenberg chain.

CPU only, per the benchmark plan (TeNPy has no GPU backend).
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tenpy.algorithms import dmrg
from tenpy.models.spins import SpinChain
from tenpy.networks.mps import MPS

from common.metrics import (
    CSVResultWriter, StepMeasurement, StepTimeoutError, completed_keys, cpu_timed_block, time_limit,
)
from common.model import HEISENBERG_J, N_SWEEPS, STEP_TIMEOUT_SEC, param_grid


def run_one(chi, L, dmrg_chi_max=None):
    with cpu_timed_block() as r:
        model_params = dict(
            L=L, S=0.5, Jx=HEISENBERG_J, Jy=HEISENBERG_J, Jz=HEISENBERG_J,
            bc_MPS="finite", conserve=None,
        )
        M = SpinChain(model_params)
        product_state = (["up", "down"] * (L // 2 + 1))[:L]
        psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)

        dmrg_params = {
            "mixer": True,
            "trunc_params": {"chi_max": dmrg_chi_max or chi, "svd_min": 1e-10},
            "max_sweeps": N_SWEEPS,
            "combine": True,
        }
        eng = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)

        t0 = time.perf_counter()
        E, psi = eng.run()
        loop_time = time.perf_counter() - t0
    n_sweeps = eng.sweep_stats["sweep"][-1] if eng.sweep_stats["sweep"] else N_SWEEPS
    step_time = loop_time / max(1, n_sweeps)
    return step_time, r["peak_mem_mb"], E


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
            print(f"[tenpy/dmrg_dense] chi={chi} L={L} skipped (exceeded {STEP_TIMEOUT_SEC}s)")
            continue
        writer.write(StepMeasurement(
            library="tenpy", algorithm="dmrg_dense", symmetry="dense",
            device="cpu", backend="numpy", L=L, chi=chi,
            step_time_sec=step_time, peak_mem_mb=peak_mem_mb, answer=energy,
        ))
        print(f"[tenpy/dmrg_dense] chi={chi} L={L} "
              f"time/sweep={step_time:.4f}s peak_mem={peak_mem_mb:.1f}MB energy={energy:.6f}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results/tenpy_dmrg_dense.csv"
    main(out)
