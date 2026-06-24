"""quimb benchmark, algorithm class 4: variational ground-state search by
gradient descent on the MPS tensors of the 1D Heisenberg chain, using real
automatic differentiation of the Rayleigh quotient

    E(psi) = <psi|H|psi> / <psi|psi>

with respect to every MPS tensor simultaneously. Run with `--backend jax` or
`--backend torch` (default: both, one after another) per the requirement to
exercise quimb's AD-based optimization on both array backends.

This is quimb's natural counterpart to the manual analytic gradient used in
the TeNPy (`variational_manual_grad.py`) and Cytnx
(`variational_manual_grad.py`) benchmarks: those two libraries have no
autodiff backend, so they evaluate the closed-form gradient
`dE/dA_i* = 2*(H_eff,i(A_i) - E*A_i)` by hand. quimb's MPS/MPO tensors are
plain JAX/PyTorch arrays under the hood (via autoray dispatch), so here we
let the backend's own autodiff differentiate straight through the full
`<psi|H|psi>` and `<psi|psi>` tensor-network contractions instead.

GPU code paths are written for both backends (`device="cuda"` placement)
but cannot be exercised in this environment (no GPU).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import quimb.tensor as qtn

from common.metrics import (
    CSVResultWriter, StepMeasurement, StepTimeoutError, completed_keys, cpu_timed_block,
    jax_gpu_timed_block, time_limit, torch_gpu_timed_block,
)
from common.model import HEISENBERG_J, N_GRAD_STEPS, STEP_TIMEOUT_SEC, param_grid

LEARNING_RATE = 1e-3
DEVICE = "cpu"  # set to "gpu" to exercise the (untested) GPU code paths below


def _build(chi, L):
    psi = qtn.MPS_rand_state(L, bond_dim=chi, dtype="float64", seed=0)
    H = qtn.MPO_ham_heis(L, j=HEISENBERG_J, cyclic=False)
    return psi, H


def run_one_jax(chi, L):
    import jax
    import jax.numpy as jnp

    psi, H = _build(chi, L)
    if DEVICE == "gpu":
        device = jax.devices("gpu")[0]
    else:
        device = jax.devices("cpu")[0]
    arrays = tuple(jax.device_put(jnp.asarray(a), device) for a in psi.arrays)
    H.apply_to_arrays(lambda x: jax.device_put(jnp.asarray(x), device))

    def energy(arrays):
        p = psi.copy()
        for i, a in enumerate(arrays):
            p[i].modify(data=a)
        num = p.H @ (H.apply(p))
        den = p.H @ p
        return jnp.real(num / den)

    def norm_sq(arrays):
        p = psi.copy()
        for i, a in enumerate(arrays):
            p[i].modify(data=a)
        return jnp.real(p.H @ p)

    grad_fn = jax.jit(jax.grad(energy)) if DEVICE == "cpu" else jax.grad(energy)
    timed_block = jax_gpu_timed_block if DEVICE == "gpu" else cpu_timed_block

    def grad_step(arrays):
        g = grad_fn(arrays)
        new_arrays = []
        for a, ga in zip(arrays, g):
            gnorm = jnp.linalg.norm(ga)
            direction = jnp.where(gnorm > 1e-12, ga / gnorm, ga)
            a_new = a - LEARNING_RATE * direction
            new_arrays.append(a_new)
        # Rescale the whole state by a single global factor derived from
        # <psi|psi>, distributed evenly across all L tensors, rather than
        # normalizing each tensor independently -- the MPS is not in
        # canonical form here, so per-tensor normalization does not keep
        # the contracted <psi|psi> close to 1.
        scale = norm_sq(tuple(new_arrays)) ** (-1.0 / (2 * len(new_arrays)))
        new_arrays = [a * scale for a in new_arrays]
        return tuple(new_arrays)

    with timed_block() as r:
        for _ in range(N_GRAD_STEPS):
            arrays = grad_step(arrays)
    step_time = r["time_sec"] / N_GRAD_STEPS
    final_energy = float(energy(arrays))
    return step_time, r["peak_mem_mb"], final_energy


def run_one_torch(chi, L):
    import torch

    psi, H = _build(chi, L)
    torch_device = "cuda" if DEVICE == "gpu" else "cpu"
    arrays = [
        torch.as_tensor(a, dtype=torch.float64, device=torch_device).clone().requires_grad_(True)
        for a in psi.arrays
    ]
    H.apply_to_arrays(lambda x: torch.tensor(x, dtype=torch.float64, device=torch_device))

    def energy(arrays):
        p = psi.copy()
        for i, a in enumerate(arrays):
            p[i].modify(data=a)
        num = p.H @ (H.apply(p))
        den = p.H @ p
        e = num / den
        return torch.real(e) if torch.is_complex(e) else e

    timed_block = torch_gpu_timed_block if DEVICE == "gpu" else cpu_timed_block

    def norm_sq(arrays):
        p = psi.copy()
        for i, a in enumerate(arrays):
            p[i].modify(data=a)
        return p.H @ p

    def grad_step(arrays):
        for a in arrays:
            if a.grad is not None:
                a.grad = None
        e = energy(arrays)
        e.backward()
        new_arrays = []
        with torch.no_grad():
            for a in arrays:
                gnorm = a.grad.norm()
                direction = a.grad / gnorm if gnorm > 1e-12 else a.grad
                a_new = a - LEARNING_RATE * direction
                new_arrays.append(a_new)
            # Rescale the whole state by a single global factor derived from
            # <psi|psi>, distributed evenly across all L tensors, rather than
            # normalizing each tensor independently -- the MPS is not in
            # canonical form here, so per-tensor normalization does not keep
            # the contracted <psi|psi> close to 1.
            scale = norm_sq(new_arrays) ** (-1.0 / (2 * len(new_arrays)))
            new_arrays = [(a * scale).clone().requires_grad_(True) for a in new_arrays]
        return new_arrays

    with timed_block() as r:
        for _ in range(N_GRAD_STEPS):
            arrays = grad_step(arrays)
    step_time = r["time_sec"] / N_GRAD_STEPS
    with torch.no_grad():
        final_energy = float(energy(arrays))
    return step_time, r["peak_mem_mb"], final_energy


def main(out_csv, backends):
    writer = CSVResultWriter(out_csv)
    done = completed_keys(out_csv, "backend", "chi", "L")
    runners = {"jax": run_one_jax, "torch": run_one_torch}
    for backend in backends:
        run_one = runners[backend]
        for chi, L in param_grid():
            if (backend, str(chi), str(L)) in done:
                continue
            try:
                with time_limit(STEP_TIMEOUT_SEC):
                    step_time, peak_mem_mb, energy_val = run_one(chi, L)
            except StepTimeoutError:
                print(f"[quimb/variational_ad/{backend}] chi={chi} L={L} "
                      f"skipped (exceeded {STEP_TIMEOUT_SEC}s)")
                continue
            writer.write(StepMeasurement(
                library="quimb", algorithm="variational_ad", symmetry="dense",
                device=DEVICE, backend=backend, L=L, chi=chi,
                step_time_sec=step_time, peak_mem_mb=peak_mem_mb, answer=energy_val,
            ))
            print(f"[quimb/variational_ad/{backend}] chi={chi} L={L} "
                  f"time/step={step_time:.4f}s peak_mem={peak_mem_mb:.1f}MB energy={energy_val:.6f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("out_csv", nargs="?", default="results/quimb_variational_ad.csv")
    parser.add_argument("--backend", choices=["jax", "torch", "both"], default="both")
    args = parser.parse_args()
    backends = ["jax", "torch"] if args.backend == "both" else [args.backend]
    main(args.out_csv, backends)
