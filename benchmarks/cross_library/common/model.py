"""Shared physical model and parameter-grid definitions for the
TeNPy / quimb / Cytnx cross-library benchmark suite.

All four algorithm classes benchmark the *same* physical model and the
*same* (chi, L) parameter grid, so that the only thing that varies between
runs is the library/implementation:

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

# 2D sweep grid shared by every algorithm/library: bond dimension chi vs.
# chain length L. This is the axis pair the user asked to scan in order to
# expose the memory (~O(L * chi^2) for an MPS) and speed (~O(L * chi^3) for
# a DMRG/TDVP local update) tradeoffs.
CHI_VALUES = [16, 32, 64, 128, 256]
L_VALUES = [20, 50, 100, 200]

# Number of DMRG sweeps / gradient steps measured per (chi, L) point. Kept
# small because we only need a handful of steps to get a stable per-step
# timing and peak-memory estimate, not a converged ground state.
N_SWEEPS = 3
N_GRAD_STEPS = 20

# Number of Lanczos iterations for the local two-site eigensolver, shared
# between the Cytnx and quimb dense-DMRG implementations.
LANCZOS_MAXITER = 4


def param_grid():
    """Yield every (chi, L) pair in the shared sweep grid."""
    for L in L_VALUES:
        for chi in CHI_VALUES:
            yield chi, L
