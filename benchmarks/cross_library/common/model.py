"""Shared physical model and parameter-grid definitions for the
TeNPy / quimb / Cytnx cross-library benchmark suite.

All four algorithm classes benchmark the *same* physical model and the
*same* (bond_dim, num_sites) parameter grid, so that the only thing that
varies between runs is the library/implementation:

  - DMRG (dense)            : 1D spin-1/2 Heisenberg chain, no symmetry
  - DMRG (block-sparse)     : same chain, U(1) total-Sz conserved
  - TEBD/TDVP (dynamics)    : 1D transverse-field Ising chain, field quench
  - Variational AD/manual   : same Heisenberg chain as the dense DMRG case
"""

# Heisenberg coupling J (isotropic, antiferromagnetic) for DMRG / variational.
HEISENBERG_J = 1.0

# Transverse-field Ising parameters for the TEBD/TDVP quench benchmark.
# H(t<0) = -J * sum ZZ - hx_i * sum X   (paramagnetic ground state)
# H(t>=0) = -J * sum ZZ - hx_f * sum X  (quench drives entanglement growth)
TFIM_J = 1.0
TFIM_HX_INITIAL = 2.0
TFIM_HX_FINAL = 0.5
TFIM_DT = 0.05
TFIM_N_STEPS = 40

# 2D sweep grid shared by every algorithm/library: bond dimension vs.
# number of sites. This is the axis pair the user asked to scan in order to
# expose the memory (~O(num_sites * bond_dim^2) for an MPS) and speed
# (~O(num_sites * bond_dim^3) for a DMRG/TDVP local update) tradeoffs.
BOND_DIM_VALUES = [16, 32, 64]
NUM_SITES_VALUES = [20, 30, 50]

# Number of DMRG sweeps measured per (bond_dim, num_sites) point. Kept small
# because we only need a handful of sweeps to get a stable per-sweep timing
# and peak-memory estimate, not a converged ground state.
N_SWEEPS = 3

# Number of Lanczos iterations for Cytnx's local two-site eigensolver
# (cytnx_bench/test_dmrg_{dense,symmetric}.py). quimb's DMRG2 and TeNPy's
# TwoSiteDMRGEngine run their own local eigensolver to its native default
# convergence instead (quimb's scipy-eigsh-backed solver has no maxiter set
# by default; TeNPy's Lanczos defaults to N_max=20) -- neither library is
# capped to match this value. This asymmetry is intentional: each library is
# left to use its own local-eigensolve convergence behavior rather than
# forcing an artificial cap that has no natural meaning outside Cytnx's
# Lanczos call, and validate_correctness.py's --generate-references mode
# confirms all three still converge to the same ground-state energy despite
# the differing per-bond eigensolver budgets.
LANCZOS_MAXITER = 4

# Per-(bond_dim, num_sites) wall-clock budget. Points that exceed this are
# skipped rather than measured, so a handful of slow large-bond_dim/
# large-num_sites points don't dominate the suite's total run time.
GRID_POINT_TIMEOUT_SEC = 120
