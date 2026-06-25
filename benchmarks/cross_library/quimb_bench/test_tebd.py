"""quimb benchmark, algorithm class 2: real-time evolution of a 1D
transverse-field Ising chain after a field quench, using quimb's built-in
`TEBD` engine.

CPU and GPU code paths are both written; the GPU path moves the initial
MPS to the device array backend before stepping. It cannot be exercised
in this environment (no GPU).

Run timing with `pytest --benchmark-only test_tebd.py`, memory with
`pytest --memray test_tebd.py`.
"""
import pytest

import quimb.tensor as qtn

from common.model import CHI_VALUES, L_VALUES, STEP_TIMEOUT_SEC, TFIM_DT, TFIM_HX_FINAL, TFIM_J, TFIM_N_STEPS

DEVICE = "cpu"  # set to "gpu" to exercise the (untested) GPU code path below

REFERENCE_ENERGIES = {
    (16, 20): 4.750010689839629,
    (16, 30): 7.25001682418871,
    (16, 50): 12.250029014561504,
    (32, 20): 4.750010689839629,
    (32, 30): 7.25001682418871,
    (32, 50): 12.250029014561504,
    (64, 20): 4.750010689839629,
    (64, 30): 7.25001682418871,
    (64, 50): 12.250029014561504,
}


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
    for _ in range(TFIM_N_STEPS):
        tebd.step(order=2, dt=TFIM_DT)
    H_mpo = qtn.MPO_ham_ising(L, j=TFIM_J, bx=TFIM_HX_FINAL, cyclic=False)
    energy = tebd.pt.H @ (H_mpo.apply(tebd.pt))
    return float(energy.real)


@pytest.mark.timeout(STEP_TIMEOUT_SEC)
@pytest.mark.parametrize("length", L_VALUES)
@pytest.mark.parametrize("chi", CHI_VALUES)
def test_tebd_benchmark(benchmark, chi, length):
    energy = benchmark.pedantic(run_one, args=(chi, length), rounds=1, iterations=1)
    benchmark.extra_info["energy"] = energy
    assert energy == pytest.approx(REFERENCE_ENERGIES[(chi, length)], rel=1e-6)


@pytest.mark.limit_memory("60 MB")
def test_tebd_memory():
    energy = run_one(16, 20)
    assert energy == pytest.approx(REFERENCE_ENERGIES[(16, 20)], rel=1e-6)
