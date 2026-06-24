"""TeNPy benchmark, algorithm class 2: real-time evolution of a 1D
transverse-field Ising chain after a field quench, using TEBD.

(TeNPy also ships a TDVP engine with an almost identical call signature;
TEBD is used here because it is the simpler, more universally-supported
entanglement-growth benchmark and keeps the same comparison point across
all three libraries. See the project report's algorithm-class 2 for the
TDVP variant, which would only change the `tebd.TEBDEngine` line below to
`tdvp.TDVPEngine`.) CPU only.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tenpy.algorithms import tebd
from tenpy.models.tf_ising import TFIChain
from tenpy.networks.mps import MPS

from common.metrics import CSVResultWriter, StepMeasurement, StepTimeoutError, cpu_timed_block, time_limit
from common.model import STEP_TIMEOUT_SEC, TFIM_DT, TFIM_HX_FINAL, TFIM_J, TFIM_N_STEPS, param_grid


def run_one(chi, L):
    model_params = dict(L=L, J=TFIM_J, g=TFIM_HX_FINAL, bc_MPS="finite", conserve=None)
    M = TFIChain(model_params)
    # Start fully polarized along x (paramagnetic ground state of the
    # pre-quench Hamiltonian at large field) then quench to g=TFIM_HX_FINAL.
    psi = MPS.from_product_state(M.lat.mps_sites(), ["up"] * L, bc=M.lat.bc_MPS)

    tebd_params = {
        "N_steps": 1,
        "dt": TFIM_DT,
        "order": 2,
        "trunc_params": {"chi_max": chi, "svd_min": 1e-10},
    }
    eng = tebd.TEBDEngine(psi, M, tebd_params)

    with cpu_timed_block() as r:
        for _ in range(TFIM_N_STEPS):
            eng.run()
    step_time = r["time_sec"] / TFIM_N_STEPS
    energy = M.H_MPO.expectation_value(psi)
    return step_time, r["peak_mem_mb"], energy


def main(out_csv):
    writer = CSVResultWriter(out_csv)
    for chi, L in param_grid():
        try:
            with time_limit(STEP_TIMEOUT_SEC):
                step_time, peak_mem_mb, energy = run_one(chi, L)
        except StepTimeoutError:
            print(f"[tenpy/tebd_quench] chi={chi} L={L} skipped (exceeded {STEP_TIMEOUT_SEC}s)")
            continue
        writer.write(StepMeasurement(
            library="tenpy", algorithm="tebd_quench", symmetry="dense",
            device="cpu", backend="numpy", L=L, chi=chi,
            step_time_sec=step_time, peak_mem_mb=peak_mem_mb, answer=energy,
        ))
        print(f"[tenpy/tebd_quench] chi={chi} L={L} "
              f"time/step={step_time:.4f}s peak_mem={peak_mem_mb:.1f}MB energy={energy:.6f}")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results/tenpy_tebd.csv"
    main(out)
