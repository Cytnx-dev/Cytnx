# Cross-library tensor-network benchmark suite

Compares **TeNPy**, **quimb**, and **Cytnx** on the same four classes of
tensor-network algorithm, using the same physical models and the same
`(bond_dim, num_sites)` parameter grid for every library, so that speed and
peak-memory differences reflect implementation/library choices rather than
differences in workload.

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
`(BOND_DIM_VALUES, NUM_SITES_VALUES)` grid).

### Gradient computation in class 3

All three libraries run the same optimization algorithm: one gradient step
of the Rayleigh quotient `E(psi) = <psi|H|psi>/<psi|psi>` is taken on every
MPS tensor simultaneously (no orthogonality center, no per-step
canonicalization), followed by a single global rescale derived from the
new `<psi|psi>` and distributed evenly across all `L` tensors. Only the
gradient-computation method differs.

quimb has a native autodiff path (JAX or PyTorch arrays under the hood), so
`quimb_bench/test_variational_ad.py` differentiates `<psi|H|psi>/<psi|psi>`
directly through the backend's `grad`/`backward`.

TeNPy and Cytnx have no autodiff backend. Each gets its own hand-derived
analytic gradient of the same Rayleigh quotient with respect to every MPS
tensor `A_i` simultaneously:

```
dE/dA_i* = (2 / den) * (H_eff,i(A_i) - E * N_eff,i(A_i))
```

where `den = <psi|psi>`, `H_eff,i` is the effective one-site Hamiltonian
built from the L/R H-boundary environments around site `i`, and `N_eff,i`
is the analogous contraction with trivial (no-MPO) norm-boundary
environments. The norm-environment term is necessary because, with every
tensor updated at once and no canonicalization step, the tensors away from
site `i` are not isometric, so `N_eff,i` does not collapse to `A_i` as it
would in a one-site sweep. All four environment sets (H-left, H-right,
norm-left, norm-right) are rebuilt from scratch every gradient step, since
every tensor changes simultaneously. The TeNPy and Cytnx implementations of
this formula are written independently (`np_conserved` contractions vs.
Cytnx `UniTensor`/`Network`/`Contract`), not via a shared abstraction, since
the point of the benchmark is to compare each library's own primitives.

This whole-network update is a weaker optimizer than a one-site sweep, so
its converged energy is sensitive to each library's own initial-state
construction and RNG; correctness is validated by checking that each
library's own implementation lands close to its own reference energy
(tight per-library tolerance), not by comparing energies across libraries
or by exact-diagonalization matching (unlike classes 1 and 2, which are
ED-validated — see the docstrings in `test_dmrg_dense.py`/`test_tebd.py`
for details).

## Parameter grid

`common/model.py` defines the `(bond_dim, num_sites)` grid shared by every
script:

```
BOND_DIM_VALUES  = [16, 32, 64]
NUM_SITES_VALUES = [20, 30, 50]
```

Each `test_<name>.py` parametrizes its benchmark test over the full
Cartesian product of `BOND_DIM_VALUES` and `NUM_SITES_VALUES` (9 points), via
two stacked `@pytest.mark.parametrize` decorators — one over `bond_dim`, one
over `num_sites`. Every point is bounded by the per-point wall-clock budget
`GRID_POINT_TIMEOUT_SEC` (120s by default, enforced via `pytest-timeout`), so
a single slow large-bond_dim/large-num_sites point fails on its own rather
than hanging the rest of the run — see "pytest-benchmark / pytest-memray
regression tests" below for how to run an individual point.

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
its `test_<name>.py` file, exercised across the full `(bond_dim, num_sites)`
grid. There is no separate orchestration script and no standalone algorithm
modules.

```sh
pip install tenpy quimb cytnx jax torch
pip install -e '.[benchmark]'   # pytest-benchmark, pytest-memray, pytest-timeout

cd benchmarks/cross_library

# Full (bond_dim, num_sites) grid, timing only (skips the memory tests),
# with a pytest.approx assertion against a precomputed reference energy at
# every point:
python3 -m pytest --benchmark-only -q

# Memory only: --memray instruments every collected test, so select just
# the memory tests with the cytnx_memory marker (registered in
# pyproject.toml, applied to every *_memory test in this directory):
python3 -m pytest --memray -m cytnx_memory -q

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
the full `(bond_dim, num_sites)` grid through `pytest-benchmark`'s
`benchmark.pedantic`, asserting the returned energy against a
`REFERENCE_ENERGIES[(bond_dim, num_sites)]` dict via `pytest.approx`, so a
wrong physical answer fails the test rather than silently shipping a bad
timing number. The result is also recorded via `benchmark.extra_info["energy"]`;
pass `--benchmark-json=out.json` to capture it (and the timing statistics)
for every point in one file.

The same file's `test_<name>_memory` test takes no parameters — it calls
`run_one` directly (no `benchmark` fixture, so pytest-benchmark's overhead
doesn't contaminate memray's allocation trace) at the single canonical
`(chi=16, L=20)` point, under a `@pytest.mark.limit_memory(...)` decorator
tuned to that point's footprint.
