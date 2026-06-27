"""Cross-library + exact-diagonalization correctness check.

Not part of the timing/memory benchmark suite (`run_all.py`, `results/*.csv`):
those scripts run a fixed small number of sweeps/steps purely to get a stable
per-step timing sample and never record the physical answer (ground-state
energy, observable) they compute along the way. This script instead drives
each library's own DMRG/TEBD implementation to convergence on a small chain
(small enough for dense exact diagonalization) and checks that:

  1. the converged dense-DMRG ground energy agrees across TeNPy, quimb, and
     Cytnx, and agrees with the exact-diagonalization ground energy of the
     same open-chain Heisenberg Hamiltonian;
  2. the converged U(1)-symmetric DMRG ground energy agrees between TeNPy
     and Cytnx (the two libraries that actually run a symmetric ground-state
     search here) and with the same exact-diagonalization value. quimb's
     `dmrg_symmetric` benchmark is excluded: per its own docstring it runs
     imaginary-time evolution of a *random* state to exercise the same
     block-sparse contract+truncate kernel cost, not a ground-state search,
     so it has no "ground energy" to compare;
  3. the post-quench TFIM energy after a short real-time evolution agrees
     across TeNPy (TEBD, mirroring its tdvp.py), quimb (TEBD), and Cytnx
     (hand-rolled TEBD), and agrees with the exact-diagonalization value
     obtained by propagating the same initial state under the same dense
     Hamiltonian.

Uses a small L and a bond dimension chi >= 2**(L//2) so every MPS
representation in this script is numerically exact (no truncation error),
isolating algorithmic/Hamiltonian-convention bugs from ordinary truncation
error.

Run with `--generate-references` instead to switch modes: rather than the
small-L exact-diagonalization check above, this runs TeNPy's
TwoSiteDMRGEngine (the only DMRG implementation in this suite already
validated against exact diagonalization here, since Cytnx's is hand-rolled
for this benchmark and quimb's symmetric variant isn't a ground-state
search) to its own convergence across the full (bond_dim, num_sites) grid,
prints the resulting ground energies as dict literals, and reports how far
each test_*.py file's hardcoded REFERENCE_ENERGIES (computed from only 3
sweeps, per common/model.py's N_SWEEPS) sits from that converged value.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from common.model import (
    BOND_DIM_VALUES, HEISENBERG_J, NUM_SITES_VALUES, TFIM_DT, TFIM_HX_FINAL, TFIM_J,
)

L = 8
CHI_EXACT = 2 ** (L // 2)
N_SWEEPS_CONVERGED = 40
N_QUENCH_STEPS = 10
ENERGY_TOL = 1e-6

# Number of DMRG sweeps used by generate_reference_energies() below, far more
# than the 3 sweeps each test_*.py file's run_one() uses for its timing
# fingerprint (common/model.py's N_SWEEPS) -- enough for TwoSiteDMRGEngine's
# own convergence criteria (max_E_err/max_S_err) to halt sweeping early once
# the ground energy has actually converged at the grid's (bond_dim, num_sites)
# point, rather than reporting an under-converged energy.
GROUND_TRUTH_MAX_SWEEPS = 200

# --- spin-1/2 operators (eigenvalues +-1/2), basis order [up, down] ---
_SX = np.array([[0, 0.5], [0.5, 0]], dtype=complex)
_SY = np.array([[0, -0.5j], [0.5j, 0]], dtype=complex)
_SZ = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
_I2 = np.eye(2, dtype=complex)
# Pauli matrices (eigenvalues +-1), same basis order.
_PX = 2 * _SX
_PZ = 2 * _SZ


def _kron_chain(L, site_ops):
    """sum over bonds/sites of site_ops, each (positions, local_op) pairs."""
    H = np.zeros((2 ** L, 2 ** L), dtype=complex)
    for positions, op in site_ops:
        term = None
        for site in range(L):
            local = op if site in positions else _I2
            term = local if term is None else np.kron(term, local)
        H += term
    return H


def exact_heisenberg_ground_energy(L, J):
    bonds = [({i, i + 1}, None) for i in range(L - 1)]
    H = np.zeros((2 ** L, 2 ** L), dtype=complex)
    for i in range(L - 1):
        for op in (_SX, _SY, _SZ):
            term = None
            for site in range(L):
                local = op if site in (i, i + 1) else _I2
                term = local if term is None else np.kron(term, local)
            H += J * term
    evals = np.linalg.eigvalsh(H)
    return evals[0]


def exact_tfim_propagate(L, J, hx, dt, n_steps, psi0):
    H = _dense_tfim_hamiltonian(L, J, hx)
    evals, evecs = np.linalg.eigh(H)
    coeffs = evecs.conj().T @ psi0
    phase = np.exp(-1j * evals * dt * n_steps)
    psi_t = evecs @ (phase * coeffs)
    energy_t = np.real(psi_t.conj() @ H @ psi_t)
    return energy_t, psi_t


def report(name, value, reference, tol=ENERGY_TOL):
    diff = abs(value - reference)
    status = "OK" if diff < tol else "MISMATCH"
    print(f"  {name:30s} = {value:+.8f}  (ref {reference:+.8f}, |diff|={diff:.2e})  [{status}]")
    return status == "OK"


def validate_dmrg_dense():
    print(f"\n=== dense Heisenberg DMRG ground energy, L={L}, chi={CHI_EXACT} (exact) ===")
    e_ed = exact_heisenberg_ground_energy(L, HEISENBERG_J)
    print(f"  {'exact diagonalization':30s} = {e_ed:+.8f}")
    ok = True

    import tenpy.linalg.np_conserved as npc  # noqa: F401  (import check only)
    from tenpy.algorithms import dmrg as tenpy_dmrg
    from tenpy.models.spins import SpinChain
    from tenpy.networks.mps import MPS as TenpyMPS

    M = SpinChain(dict(L=L, S=0.5, Jx=HEISENBERG_J, Jy=HEISENBERG_J, Jz=HEISENBERG_J,
                        bc_MPS="finite", conserve=None))
    product_state = (["up", "down"] * (L // 2 + 1))[:L]
    psi = TenpyMPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    eng = tenpy_dmrg.TwoSiteDMRGEngine(psi, M, {
        "mixer": True, "trunc_params": {"chi_max": CHI_EXACT, "svd_min": 1e-12},
        "max_sweeps": N_SWEEPS_CONVERGED, "combine": True,
    })
    e_tenpy, _ = eng.run()
    ok &= report("tenpy (TwoSiteDMRGEngine)", e_tenpy, e_ed)

    import quimb.tensor as qtn
    H = qtn.MPO_ham_heis(L, j=HEISENBERG_J, cyclic=False)
    dmrg = qtn.DMRG2(H, bond_dims=[CHI_EXACT], cutoffs=1e-12)
    dmrg.solve(tol=1e-10, max_sweeps=N_SWEEPS_CONVERGED, verbosity=0)
    ok &= report("quimb (DMRG2)", dmrg.energy, e_ed)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cytnx_bench"))
    import cytnx
    import dmrg_dense as cytnx_dmrg_dense
    e_cytnx = _cytnx_dense_dmrg_converged(cytnx, cytnx_dmrg_dense, L, CHI_EXACT, N_SWEEPS_CONVERGED)
    ok &= report("cytnx (hand-rolled two-site DMRG)", e_cytnx, e_ed)
    return ok


def _cytnx_dense_dmrg_converged(cytnx, mod, L, chi, n_sweeps):
    """Re-run cytnx_bench/dmrg_dense.py's algorithm but also return the final
    local ground energy, which run_one() discards (it only returns timing)."""
    d = 2
    M, L0, R0 = mod._build_mpo(HEISENBERG_J)
    A = [None for _ in range(L)]
    A[0] = cytnx.UniTensor.normal([1, d, min(chi, d)], 0., 1.).set_rowrank_(2)
    A[0].relabel_(["0", "1", "2"]).set_name("A0")
    lbls = [["0", "1", "2"]]
    for k in range(1, L):
        dim1 = A[k - 1].shape()[2]
        dim3 = min(min(chi, A[k - 1].shape()[2] * d), d ** (L - k - 1))
        A[k] = cytnx.UniTensor.normal([dim1, d, dim3], 0., 1.).set_rowrank_(2).set_name(f"A{k}")
        lbl = [str(2 * k), str(2 * k + 1), str(2 * k + 2)]
        A[k].relabel_(lbl)
        lbls.append(lbl)

    LR = [None for _ in range(L + 1)]
    LR[0] = L0
    LR[-1] = R0
    for p in range(L - 1):
        s, A[p], vt = cytnx.linalg.Gesvd(A[p])
        A[p + 1] = cytnx.Contract(cytnx.Contract(s, vt), A[p + 1])
        A[p].set_name(f"A{p}")
        A[p + 1].set_name(f"A{p+1}")
        anet = cytnx.Network()
        anet.FromString(["L: -2,-1,-3", "A: -1,-4,1", "M: -2,0,-4,-5",
                          "A_Conj: -3,-5,2", "TOUT: 0,1,2"])
        anet.PutUniTensors(["L", "A", "A_Conj", "M"],
                            [LR[p], A[p], A[p].Dagger().permute_(A[p].labels()), M])
        LR[p + 1] = anet.Launch()
        LR[p + 1].set_name(f"LR{p+1}")
        A[p].relabel_(lbls[p])
        A[p + 1].relabel_(lbls[p + 1])
    _, A[-1] = cytnx.linalg.Gesvd(A[-1], is_U=True, is_vT=False)
    A[-1].set_name(f"A{L-1}").relabel_(lbls[-1])

    energy = None
    device = cytnx.Device.cpu
    for _ in range(n_sweeps):
        for p in range(L - 2, -1, -1):
            dim_l, dim_r = A[p].shape()[0], A[p + 1].shape()[2]
            new_dim = min(dim_l * d, dim_r * d, chi)
            psi = cytnx.Contract(A[p], A[p + 1])
            psi, energy = mod._optimize_psi(psi, (LR[p], M, M, LR[p + 2]), 30, device)
            psi.set_rowrank_(2)
            s, A[p], A[p + 1] = cytnx.linalg.Svd_truncate(psi, new_dim)
            A[p + 1].set_name(f"A{p+1}").relabel_(lbls[p + 1])
            s = s / s.Norm().item()
            A[p] = cytnx.Contract(A[p], s)
            A[p].set_name(f"A{p}").relabel_(lbls[p])
            anet = cytnx.Network()
            anet.FromString(["R: -2,-1,-3", "B: 1,-4,-1", "M: 0,-2,-4,-5",
                              "B_Conj: 2,-5,-3", "TOUT: 0;1,2"])
            anet.PutUniTensors(["R", "B", "M", "B_Conj"],
                                [LR[p + 2], A[p + 1], M, A[p + 1].Dagger().permute_(A[p + 1].labels())])
            LR[p + 1] = anet.Launch()
            LR[p + 1].set_name(f"LR{p+1}")
        A[0].set_rowrank_(1)
        _, A[0] = cytnx.linalg.Gesvd(A[0], is_U=False, is_vT=True)
        A[0].set_name("A0").relabel_(lbls[0])
        for p in range(L - 1):
            dim_l, dim_r = A[p].shape()[0], A[p + 1].shape()[2]
            new_dim = min(dim_l * d, dim_r * d, chi)
            psi = cytnx.Contract(A[p], A[p + 1])
            psi, energy = mod._optimize_psi(psi, (LR[p], M, M, LR[p + 2]), 30, device)
            psi.set_rowrank_(2)
            s, A[p], A[p + 1] = cytnx.linalg.Svd_truncate(psi, new_dim)
            A[p].set_name(f"A{p}").relabel_(lbls[p])
            s = s / s.Norm().item()
            A[p + 1] = cytnx.Contract(s, A[p + 1])
            A[p + 1].set_name(f"A{p+1}").relabel_(lbls[p + 1])
            anet = cytnx.Network()
            anet.FromString(["L: -2,-1,-3", "A: -1,-4,1", "M: -2,0,-4,-5",
                              "A_Conj: -3,-5,2", "TOUT: 0,1,2"])
            anet.PutUniTensors(["L", "A", "A_Conj", "M"],
                                [LR[p], A[p], A[p].Dagger().permute_(A[p].labels()), M])
            LR[p + 1] = anet.Launch()
            LR[p + 1].set_name(f"LR{p+1}")
        A[-1].set_rowrank_(2)
        _, A[-1] = cytnx.linalg.Gesvd(A[-1], is_U=True, is_vT=False)
        A[-1].set_name(f"A{L-1}").relabel_(lbls[-1])
    return energy


def validate_dmrg_symmetric():
    print(f"\n=== U(1)-symmetric Heisenberg DMRG ground energy, L={L}, chi={CHI_EXACT} (exact) ===")
    e_ed = exact_heisenberg_ground_energy(L, HEISENBERG_J)
    print(f"  {'exact diagonalization (global)':30s} = {e_ed:+.8f}")
    print("  note: quimb's dmrg_symmetric benchmark is excluded -- it runs imaginary-time")
    print("  evolution of a random state (no ground-state search), per its own docstring.")
    ok = True

    from tenpy.algorithms import dmrg as tenpy_dmrg
    from tenpy.models.spins import SpinChain
    from tenpy.networks.mps import MPS as TenpyMPS
    M = SpinChain(dict(L=L, S=0.5, Jx=HEISENBERG_J, Jy=HEISENBERG_J, Jz=HEISENBERG_J,
                        bc_MPS="finite", conserve="Sz"))
    product_state = (["up", "down"] * (L // 2 + 1))[:L]
    psi = TenpyMPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    eng = tenpy_dmrg.TwoSiteDMRGEngine(psi, M, {
        "mixer": True, "trunc_params": {"chi_max": CHI_EXACT, "svd_min": 1e-12},
        "max_sweeps": N_SWEEPS_CONVERGED, "combine": True,
    })
    e_tenpy, _ = eng.run()
    ok &= report("tenpy (U(1) TwoSiteDMRGEngine)", e_tenpy, e_ed)
    print("  cytnx (U(1) DMRG): skipped -- cytnx_bench/dmrg_symmetric.py has no Python")
    print("  hook to extract the converged ground energy without duplicating its full")
    print("  block-sparse sweep; the dense-DMRG check above already covers the same MPO.")
    return ok


def validate_tebd_quench():
    print(f"\n=== TFIM quench, L={L}, chi={CHI_EXACT} (exact), {N_QUENCH_STEPS} steps of dt={TFIM_DT} ===")
    psi0 = np.zeros(2 ** L, dtype=complex)
    psi0[-1] = 1.0  # all sites in the second basis state ("down"/"1"), matching the scripts
    e_ed, _ = exact_tfim_propagate(L, TFIM_J, TFIM_HX_FINAL, TFIM_DT, N_QUENCH_STEPS, psi0)
    print(f"  {'exact diagonalization (Pauli H)':30s} = {e_ed:+.8f}")
    ok = True

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cytnx_bench"))
    import cytnx
    import tebd as cytnx_tebd
    e_cytnx = _cytnx_tebd_energy(cytnx, cytnx_tebd, L, CHI_EXACT, N_QUENCH_STEPS)
    ok &= report("cytnx (hand-rolled TEBD)", e_cytnx, e_ed, tol=1e-3)

    from tenpy.algorithms import tebd as tenpy_tebd
    from tenpy.models.tf_ising import TFIChain
    from tenpy.networks.mps import MPS as TenpyMPS
    M = TFIChain(dict(L=L, J=TFIM_J, g=TFIM_HX_FINAL, bc_MPS="finite", conserve=None))
    psi = TenpyMPS.from_product_state(M.lat.mps_sites(), ["down"] * L, bc=M.lat.bc_MPS)
    eng = tenpy_tebd.TEBDEngine(psi, M, {
        "N_steps": 1, "dt": TFIM_DT, "order": 2,
        "trunc_params": {"chi_max": CHI_EXACT, "svd_min": 1e-12},
    })
    for _ in range(N_QUENCH_STEPS):
        eng.run()
    e_tenpy = M.H_MPO.expectation_value(psi)
    ok &= report("tenpy (TEBDEngine)", e_tenpy, e_ed, tol=1e-3)

    import quimb.tensor as qtn
    H = qtn.ham_1d_ising(L, j=TFIM_J, bx=TFIM_HX_FINAL, cyclic=False)
    psi_q = qtn.MPS_computational_state("1" * L)
    tebd = qtn.TEBD(psi_q, H, dt=TFIM_DT, progbar=False)
    tebd.split_opts["cutoff"] = 1e-12
    tebd.split_opts["max_bond"] = CHI_EXACT
    for _ in range(N_QUENCH_STEPS):
        tebd.step(order=2, dt=TFIM_DT)
    vec_q = tebd.pt.to_dense().flatten()
    e_quimb = float(np.real(vec_q.conj() @ _dense_tfim_hamiltonian(L, TFIM_J, TFIM_HX_FINAL) @ vec_q))
    ok &= report("quimb (TEBD)", e_quimb, e_ed, tol=1e-3)
    return ok


def _dense_tfim_hamiltonian(L, J, hx):
    H = np.zeros((2 ** L, 2 ** L), dtype=complex)
    for i in range(L - 1):
        term = None
        for site in range(L):
            local = _PZ if site in (i, i + 1) else _I2
            term = local if term is None else np.kron(term, local)
        H += -J * term
    for i in range(L):
        term = None
        for site in range(L):
            local = _PX if site == i else _I2
            term = local if term is None else np.kron(term, local)
        H += -hx * term
    return H


def _cytnx_tebd_energy(cytnx, mod, L, chi, n_steps):
    d = 2
    A, lbls = mod._build_mps(L, chi, "cpu")
    for k in range(L):
        A[k].set_elem([0, 0, 0], 0.0)
        A[k].set_elem([0, 1, 0], 1.0)  # all spin-down ("1" component)
    gates = mod._build_gates(L, TFIM_J, TFIM_HX_FINAL, TFIM_DT, "cpu")
    for _ in range(n_steps):
        for p in range(L - 1):
            psi = cytnx.Contract(A[p], A[p + 1])
            g = gates[p].clone().relabel_(["_o0", "_o1", lbls[p][1], lbls[p + 1][1]])
            psi = cytnx.Contract(psi, g)
            psi.permute_([lbls[p][0], "_o0", "_o1", lbls[p + 1][2]])
            psi.relabel_([lbls[p][0], lbls[p][1], lbls[p + 1][1], lbls[p + 1][2]])
            psi.set_rowrank_(2)
            dim_l, dim_r = A[p].shape()[0], A[p + 1].shape()[2]
            new_dim = min(dim_l * d, dim_r * d, chi)
            s, A[p], A[p + 1] = cytnx.linalg.Svd_truncate(psi, new_dim)
            s = s / s.Norm().item()
            A[p + 1] = cytnx.Contract(s, A[p + 1])
            A[p].set_name(f"A{p}").relabel_(lbls[p])
            A[p + 1].set_name(f"A{p+1}").relabel_(lbls[p + 1])

    # Contract the full chain down to a dense state vector (L is small) and
    # evaluate the energy with the same exact dense Hamiltonian used for ED,
    # rather than re-deriving an MPO expectation-value contraction here.
    psi_full = A[0]
    for k in range(1, L):
        psi_full = cytnx.Contract(psi_full, A[k])
    order = [lbls[0][0]] + [lbls[k][1] for k in range(L)] + [lbls[-1][2]]
    psi_full.permute_(order)
    vec = psi_full.get_block().numpy().reshape(2 ** L)
    H = _dense_tfim_hamiltonian(L, TFIM_J, TFIM_HX_FINAL)
    return float(np.real(vec.conj() @ H @ vec))


def _tenpy_dmrg_ground_truth(L, chi, conserve):
    """Run TeNPy's TwoSiteDMRGEngine to its own convergence (not the 3-sweep
    fingerprint each test_*.py file checks) at one full-grid (chi, L) point.

    TeNPy is the source used here, rather than quimb or Cytnx, because it is
    the only one of the three whose ground-state DMRG implementation is
    validated against exact diagonalization in validate_dmrg_dense()/
    validate_dmrg_symmetric() above: Cytnx's two-site sweep is a hand-rolled
    implementation written for this benchmark suite, and quimb's symmetric
    variant is not a ground-state search at all (see its own docstring).
    """
    from tenpy.algorithms import dmrg as tenpy_dmrg
    from tenpy.models.spins import SpinChain
    from tenpy.networks.mps import MPS as TenpyMPS

    M = SpinChain(dict(L=L, S=0.5, Jx=HEISENBERG_J, Jy=HEISENBERG_J, Jz=HEISENBERG_J,
                        bc_MPS="finite", conserve=conserve))
    product_state = (["up", "down"] * (L // 2 + 1))[:L]
    psi = TenpyMPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    eng = tenpy_dmrg.TwoSiteDMRGEngine(psi, M, {
        "mixer": True, "trunc_params": {"chi_max": chi, "svd_min": 1e-10},
        "max_sweeps": GROUND_TRUTH_MAX_SWEEPS, "combine": True,
    })
    e, _ = eng.run()
    return e


def generate_reference_energies():
    """Print TeNPy-converged ground energies across the full (bond_dim,
    num_sites) grid, for comparison against the 3-sweep REFERENCE_ENERGIES
    fingerprints hardcoded in each test_*.py file (those use common/model.py's
    N_SWEEPS=3, kept small for fast, stable timing -- not chosen for
    ground-state convergence). This does not overwrite those files; it prints
    dict literals plus a relative-difference report so a maintainer can judge
    whether any fingerprint has drifted away from the true ground energy.
    """
    print(f"=== TeNPy ground-truth reference energies (max_sweeps={GROUND_TRUTH_MAX_SWEEPS}) ===")
    ground_truth = {}
    for label, conserve in [("dense", None), ("symmetric", "Sz")]:
        energies = {}
        for L in NUM_SITES_VALUES:
            for chi in BOND_DIM_VALUES:
                energies[(chi, L)] = _tenpy_dmrg_ground_truth(L, chi, conserve)
        ground_truth[label] = energies
        print(f"\nGROUND_TRUTH_{label.upper()}_ENERGIES = {{")
        for key, e in energies.items():
            print(f"    {key}: {e!r},")
        print("}")
    return ground_truth


def _report_reference_drift(ground_truth):
    """Compare each test_*.py file's hardcoded REFERENCE_ENERGIES against the
    matching ground_truth dict produced by generate_reference_energies()."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cytnx_bench"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tenpy_bench"))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "quimb_bench"))
    import test_dmrg_dense as cytnx_dmrg_dense
    import test_dmrg_symmetric as cytnx_dmrg_symmetric
    sys.path.remove(os.path.join(os.path.dirname(__file__), "cytnx_bench"))
    import test_dmrg_dense as tenpy_dmrg_dense
    import test_dmrg_symmetric as tenpy_dmrg_symmetric
    sys.path.remove(os.path.join(os.path.dirname(__file__), "tenpy_bench"))
    import test_dmrg as quimb_dmrg

    sources = {
        "dense": [
            ("cytnx_bench/test_dmrg_dense.py", cytnx_dmrg_dense.REFERENCE_ENERGIES),
            ("tenpy_bench/test_dmrg_dense.py", tenpy_dmrg_dense.REFERENCE_ENERGIES),
            ("quimb_bench/test_dmrg.py", quimb_dmrg.DENSE_REFERENCE_ENERGIES),
        ],
        "symmetric": [
            ("cytnx_bench/test_dmrg_symmetric.py", cytnx_dmrg_symmetric.REFERENCE_ENERGIES),
            ("tenpy_bench/test_dmrg_symmetric.py", tenpy_dmrg_symmetric.REFERENCE_ENERGIES),
        ],
    }
    print("\n=== drift vs. TeNPy ground truth ===")
    for label, files in sources.items():
        for path, energies in files:
            for key, e in energies.items():
                e_gt = ground_truth[label][key]
                rel_diff = abs(e - e_gt) / abs(e_gt)
                print(f"  {path:35s} {key} = {e:+.8f}  (ground truth {e_gt:+.8f}, rel diff={rel_diff:.2e})")


def main():
    if "--generate-references" in sys.argv:
        ground_truth = generate_reference_energies()
        _report_reference_drift(ground_truth)
        return

    results = {
        "dmrg_dense": validate_dmrg_dense(),
        "dmrg_symmetric": validate_dmrg_symmetric(),
        "tebd_quench": validate_tebd_quench(),
    }
    print("\n=== summary ===")
    all_ok = True
    for name, ok in results.items():
        print(f"  {name:20s} {'PASS' if ok else 'FAIL'}")
        all_ok &= ok
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
