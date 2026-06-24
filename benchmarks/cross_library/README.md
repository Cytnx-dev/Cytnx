# Cross-library tensor-network benchmark suite

Compares **TeNPy**, **quimb**, and **Cytnx** on the same four classes of
tensor-network algorithm, using the same physical models and the same
`(chi, L)` parameter grid for every library, so that speed and peak-memory
differences reflect implementation/library choices rather than differences
in workload.

## Algorithm classes

| # | Class | Model | Library implementations |
|---|-------|-------|--------------------------|
| 1 | Finite two-site DMRG, dense | 1D spin-1/2 Heisenberg chain, no symmetry | `tenpy_bench/dmrg_dense.py`, `quimb_bench/dmrg_dense.py`, `cytnx_bench/dmrg_dense.py` |
| 1' | Finite two-site DMRG, block-sparse | Same chain, U(1) total-Sz conserved | `tenpy_bench/dmrg_symmetric.py`, `quimb_bench/dmrg_symmetric.py`, `cytnx_bench/dmrg_symmetric.py` |
| 2 | Real-time evolution after a field quench (TEBD/TDVP) | 1D transverse-field Ising chain | `tenpy_bench/tdvp.py`, `quimb_bench/tebd.py`, `cytnx_bench/tebd.py` |
| 3 | Variational MPS ground-state search by gradient descent | Same Heisenberg chain as class 1 | `tenpy_bench/variational_manual_grad.py`, `quimb_bench/variational_ad.py`, `cytnx_bench/variational_manual_grad.py` |

All four classes share the model/parameter definitions in `common/model.py`
(`HEISENBERG_J`, `TFIM_J`/`TFIM_HX_INITIAL`/`TFIM_HX_FINAL`/`TFIM_DT`, the
`(CHI_VALUES, L_VALUES)` grid) and the timing/memory helpers in
`common/metrics.py`.

### Gradient computation in class 3

quimb has a native autodiff path (JAX or PyTorch arrays under the hood), so
`quimb_bench/variational_ad.py` differentiates `<psi|H|psi>/<psi|psi>`
directly through the backend's `grad`/`backward`.

TeNPy and Cytnx have no autodiff backend. Each gets its own hand-derived
analytic gradient of the same Rayleigh quotient with respect to a single MPS
tensor `A_i` (all other tensors held fixed):

```
dE/dA_i* = 2 * (H_eff,i(A_i) - E * A_i)
```

where `H_eff,i` is the effective one-site Hamiltonian built from the L/R
boundary environments around site `i`. The TeNPy and Cytnx implementations
of this formula are written independently (`np_conserved` contractions vs.
Cytnx `UniTensor`/`Network`/`Contract`), not via a shared abstraction, since
the point of the benchmark is to compare each library's own primitives.
Both implementations evaluate the local Rayleigh quotient without enforcing
strict mixed-canonical gauge on neighboring tensors, so the per-site energy
is only an approximation to the global `<psi|H|psi>/<psi|psi>`; this was
validated by checking that the local energy decreases monotonically over
gradient steps, not by exact-diagonalization matching (unlike classes 1 and
2, which are ED-validated — see the docstrings in `dmrg_dense.py`/
`tebd.py` for details).

## Parameter grid

`common/model.py`'s `param_grid()` yields the full Cartesian product of:

```
CHI_VALUES = [16, 32, 64, 128, 256]
L_VALUES   = [20, 50, 100, 200]
```

Every benchmark script runs all `len(CHI_VALUES) * len(L_VALUES)` points and
writes one CSV row per point.

## CPU vs. GPU

- **TeNPy**: CPU only (TeNPy itself has no GPU backend).
- **quimb**: CPU and GPU paths are both implemented for the AD benchmark
  (`variational_ad.py --backend jax|torch|both`); the DMRG/TEBD scripts also
  carry a `DEVICE = "gpu"` switch.
- **Cytnx**: every script has a `DEVICE` module-level flag (`"cpu"` or
  `"gpu"`); the GPU path moves every MPS/MPO `UniTensor` to
  `cytnx.Device.cuda` before the sweep/step.

GPU code paths exist in every script that supports them but were written
and reviewed without a GPU available in the development environment, so
they are untested. Set `DEVICE = "gpu"` at the top of the relevant script(s)
to exercise them on a CUDA-capable machine.

## Running the suite

```sh
pip install tenpy quimb cytnx jax torch matplotlib pandas

cd benchmarks/cross_library
python run_all.py                     # run every library/algorithm, write results/*.csv
python run_all.py --only cytnx quimb  # restrict to a subset
python run_all.py --skip tenpy        # skip libraries you don't have installed

python plot_results.py                # read results/*.csv, write results/plots/*.png
```

Each benchmark script can also be run standalone, e.g.:

```sh
python cytnx_bench/dmrg_dense.py results/cytnx_dmrg_dense.csv
```

## Output format

Every script writes rows of (see `common/metrics.py:StepMeasurement`):

| field | meaning |
|---|---|
| `library` | `tenpy`, `quimb`, or `cytnx` |
| `algorithm` | `dmrg_dense`, `dmrg_symmetric`, `tebd_quench`, `variational_manual_grad`, or `variational_ad` |
| `symmetry` | `dense` or `u1` |
| `device` | `cpu` or `gpu` |
| `backend` | e.g. `cytnx`, `manual-grad`, `jax`, `torch` |
| `L`, `chi` | chain length, bond dimension for this data point |
| `step_time_sec` | wall-clock time per DMRG sweep / TEBD step / gradient step |
| `peak_mem_mb` | peak memory for that block (CPU: max of tracemalloc and RSS delta; GPU: backend's peak-allocator counter) |
