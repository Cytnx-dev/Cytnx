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

from common.model import TFIM_DT, TFIM_HX_FINAL, TFIM_J, TFIM_N_STEPS

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
    for _ in range(TFIM_N_STEPS):
        tebd.step(order=2, dt=TFIM_DT)
    H_mpo = qtn.MPO_ham_ising(L, j=TFIM_J, bx=TFIM_HX_FINAL, cyclic=False)
    energy = tebd.pt.H @ (H_mpo.apply(tebd.pt))
    return float(energy.real)
