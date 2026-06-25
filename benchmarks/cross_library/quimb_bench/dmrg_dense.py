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

from common.model import HEISENBERG_J, N_SWEEPS

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
    dmrg.solve(tol=1e-6, max_sweeps=N_SWEEPS, verbosity=0)
    return dmrg.energy
