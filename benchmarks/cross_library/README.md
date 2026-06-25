# Cross-library tensor-network benchmark suite

Compares **TeNPy**, **quimb**, and **Cytnx** on the same four classes of
tensor-network algorithm, using the same physical models and the same
`(chi, L)` parameter grid for every library, so that speed and peak-memory
differences reflect implementation/library choices rather than differences
in workload.

## Algorithm classes

| # | Class | Model | Library implementations |
|---|-------|-------|--------------------------|
| 1 | Finite two-site DMRG, dense | 1D spin-1/2 Heisenberg chain, no symmetry | `tenpy_bench/test_dmrg_dense.py`, `quimb_bench/test_dmrg.py`, `cytnx_bench/test_dmrg_dense.py` |
| 1' | Finite two-site DMRG, block-sparse | Same chain, U(1) total-Sz conserved | `tenpy_bench/test_dmrg_symmetric.py`, `quimb_bench/test_dmrg.py`, `cytnx_bench/test_dmrg_symmetric.py` |
| 2 | Real-time evolution after a field quench (TEBD/TDVP) | 1D transverse-field Ising chain | `tenpy_bench/test_tdvp.py`, `quimb_bench/test_tebd.py`, `cytnx_bench/test_tebd.py` |
| 3 | Variational MPS ground-state search by gradient descent | Same Heisenberg chain as class 1 | `tenpy_bench/test_variational_manual_grad.py`, `quimb_bench/test_variational_ad.py`, `cytnx_bench/test_variational_manual_grad.py` |

Each script's `run_one(chi, L)` lives directly in its `test_*.py` file
(there are no separate, non-test algorithm modules) so that every test
file is self-contained and importable as ordinary pytest-discovered code,
with no `sys.path` manipulation. `quimb_bench/test_dmrg.py` covers both
class 1 and class 1' in one file, since quimb's block-sparse variant
(`run_one_symmetric`) is a self-consistency check (imaginary-time
evolution from a random seeded state), not a ground-state energy
comparable across libraries — see the docstring in that file.

All four classes share the model/parameter definitions in `common/model.py`
(`HEISENBERG_J`, `TFIM_J`/`TFIM_HX_INITIAL`/`TFIM_HX_FINAL`/`TFIM_DT`, the
`(CHI_VALUES, L_VALUES)` grid).

### Gradient computation in class 3

quimb has a native autodiff path (JAX or PyTorch arrays under the hood), so
`quimb_bench/test_variational_ad.py` differentiates `<psi|H|psi>/<psi|psi>`
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
2, which are ED-validated — see the docstrings in `test_dmrg_dense.py`/
`test_tebd.py` for details).

## Parameter grid

`common/model.py` defines the `(chi, L)` grid shared by every script:

```
CHI_VALUES = [16, 32, 64]
L_VALUES   = [20, 30, 50]
```

Each `test_<name>.py` parametrizes its benchmark test over the full
Cartesian product of `CHI_VALUES` and `L_VALUES` (9 points), via two stacked
`@pytest.mark.parametrize` decorators — one over `chi`, one over `length`.
Every point is bounded by the per-point wall-clock budget `STEP_TIMEOUT_SEC`
(120s by default, enforced via `pytest-timeout`), so a single slow
large-chi/large-L point fails on its own rather than hanging the rest of the
run — see "pytest-benchmark / pytest-memray regression tests" below for how
to run an individual point.

## CPU vs. GPU

- **TeNPy**: CPU only (TeNPy itself has no GPU backend).
- **quimb**: CPU and GPU paths are both implemented for the AD benchmark
  (`variational_ad.py`'s `run_one_jax`/`run_one_torch`); the DMRG/TEBD scripts
  also carry a `DEVICE = "gpu"` switch.
- **Cytnx**: every script has a `DEVICE` module-level flag (`"cpu"` or
  `"gpu"`); the GPU path moves every MPS/MPO `UniTensor` to
  `cytnx.Device.cuda` before the sweep/step.

GPU code paths exist in every script that supports them but were written
and reviewed without a GPU available in the development environment, so
they are untested. Set `DEVICE = "gpu"` at the top of the relevant script(s)
to exercise them on a CUDA-capable machine.

## Running the suite

The whole suite is pytest-native: each script's `run_one(chi, L)` lives in
its `test_<name>.py` file, exercised across the full `(chi, L)` grid. There
is no separate orchestration script and no standalone algorithm modules.

```sh
pip install tenpy quimb cytnx jax torch
pip install -e '.[benchmark]'   # pytest-benchmark, pytest-memray, pytest-timeout

cd benchmarks/cross_library

# Full (chi, L) grid, timing only (skips the limit_memory tests), with a
# pytest.approx assertion against a precomputed reference energy at every
# point:
python3 -m pytest --benchmark-only -q

# Memory, at the single canonical (chi=16, L=20) point each limit_memory
# test is pinned to:
python3 -m pytest --memray -q

# Run the whole grid for one script:
python3 -m pytest cytnx_bench/test_dmrg_dense.py --benchmark-only -q

# Or call pytest one point at a time, which doubles as a resume mechanism if
# a run gets interrupted partway through the grid:
python3 -m pytest "cytnx_bench/test_dmrg_dense.py::test_dmrg_dense_benchmark[16-20]" --benchmark-only -q
```

Run from `benchmarks/cross_library` (not the repo root) with `python3 -m
pytest` (not the bare `pytest` command) so that `cytnx`/`quimb`/`tenpy`
resolve to your installed packages rather than colliding with the repo's
source tree.

## pytest-benchmark / pytest-memray regression tests

Each of the 12 `test_<name>.py` files exercises its `run_one(chi, L)` across
the full `(chi, L)` grid through `pytest-benchmark`'s `benchmark.pedantic`,
asserting the returned energy against a `REFERENCE_ENERGIES[(chi, length)]`
dict via `pytest.approx`, so a wrong physical answer fails the test rather
than silently shipping a bad timing number. The result is also recorded via
`benchmark.extra_info["energy"]`; pass `--benchmark-json=out.json` to capture
it (and the timing statistics) for every point in one file.

The same file's `test_<name>_memory` test takes no parameters — it calls
`run_one` directly (no `benchmark` fixture, so pytest-benchmark's overhead
doesn't contaminate memray's allocation trace) at the single canonical
`(chi=16, L=20)` point, under a `@pytest.mark.limit_memory(...)` decorator
tuned to that point's footprint.
