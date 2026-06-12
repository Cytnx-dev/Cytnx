#ifndef _H_linalg_test
#define _H_linalg_test

#include "cytnx.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <utility>
#include <vector>

using namespace cytnx;
using namespace std;

class linalg_Test : public ::testing::Test {
 public:
  // ==================== general ===================
  Tensor arange3x3d = arange(0, 9, 1, Type.Double).reshape(3, 3);
  Tensor ones3x3d = ones(9, Type.Double).reshape(3, 3);
  Tensor eye3x3d = eye(3, Type.Double);
  Tensor zeros3x3d = zeros(9, Type.Double).reshape(3, 3);

  Tensor arange3x3cd = arange(0, 9, 1, Type.ComplexDouble).reshape(3, 3) +
                       cytnx_complex128(0, 1) * arange(0, 9, 1, Type.ComplexDouble).reshape(3, 3);
  Tensor ones3x3cd = ones(9, Type.ComplexDouble).reshape(3, 3);
  Tensor eye3x3cd = eye(3, Type.ComplexDouble);
  Tensor zeros3x3cd = zeros(9, Type.ComplexDouble).reshape(3, 3);

  Tensor invertable3x3cd = arange(1, 10, 1, Type.ComplexDouble).reshape(3, 3);

  UniTensor arange3x3cd_ut = UniTensor(arange3x3cd, false, -1);
  UniTensor ones3x3cd_ut = UniTensor(ones3x3cd, false, -1);
  UniTensor invertable3x3cd_ut = UniTensor(invertable3x3cd, false, -1);

  std::string data_dir = CYTNX_TEST_DATA_DIR "/linalg/";
  // ==================== svd_truncate ===================
  Bond svd_I = Bond(BD_OUT, {Qs(1), Qs(-1)}, {1, 1});
  Bond svd_J =
    Bond(BD_IN, {Qs(5), Qs(3), Qs(1), Qs(-1), Qs(-3), Qs(-5), Qs(-7)}, {6, 22, 57, 68, 38, 8, 1});
  Bond svd_K =
    Bond(BD_OUT, {Qs(5), Qs(3), Qs(1), Qs(-1), Qs(-3), Qs(-5), Qs(-7)}, {6, 22, 57, 68, 38, 8, 1});
  Bond svd_L = Bond(BD_OUT, {Qs(1), Qs(-1)}, {1, 1});
  UniTensor svd_T = UniTensor({svd_I, svd_J, svd_K, svd_L}, {"a", "b", "c", "d"}, 1, Type.Double,
                              Device.cpu, false);

  UniTensor svd_T_dense =
    UniTensor(arange(0, 11 * 13, 1).reshape(11, 13)).astype(Type.ComplexDouble).to(Device.cpu);
  Tensor svd_Sans;
  //==================== Lanczos_Gnd_Ut ===================
  Tensor A = Tensor::Load(data_dir + "Lanczos_Gnd/lan_block_A.cytn");
  Tensor B = Tensor::Load(data_dir + "Lanczos_Gnd/lan_block_B.cytn");
  Tensor C = Tensor::Load(data_dir + "Lanczos_Gnd/lan_block_C.cytn");
  Bond lan_I = Bond(BD_IN, {Qs(-1), Qs(0), Qs(1)}, {9, 9, 9});
  Bond lan_J = Bond(BD_OUT, {Qs(-1), Qs(0), Qs(1)}, {9, 9, 9});
  UniTensor H = UniTensor({lan_I, lan_J});

  //==================== QR ===================
  Tensor Qr_Qans = Tensor::Load(data_dir + "Qr/qr_Qans.cytn");
  Tensor Qr_Rans = Tensor::Load(data_dir + "Qr/qr_Rans.cytn");

  //==================== ExpH ===================
  Tensor expH_ans = Tensor::Load(data_dir + "expH/expH_ans.cytn");

  //==================== Pow ===================
  Tensor Pow_ans = Tensor::Load(data_dir + "Pow/Pow_ans.cytn");

  //==================== Mod ===================
  Tensor Mod_ans = Tensor::Load(data_dir + "Mod/Mod_ans.cytn");
  Tensor ModUtUt_ans = Tensor::Load(data_dir + "Mod/ModUtUt_ans.cytn");

 protected:
  void SetUp() override {
    //================ svd truncate =======================
    svd_T = svd_T.Load(data_dir + "Svd_truncate/Svd_truncate1.cytnx");
    svd_T.permute_({1, 0, 3, 2});
    svd_T.contiguous_();
    svd_T.set_rowrank_(2);
    svd_Sans = Tensor::Load(data_dir + "Svd_truncate/S_truncate1.cytn");
    svd_Sans = algo::Sort(svd_Sans);
    //==================== Lanczos_Gnd_Ut ===================
    H.put_block(A, 0);
    H.put_block(B, 1);
    H.put_block(C, 2);
    H.relabel_({"a", "b"});

    invertable3x3cd.at({0, 0}) = 2;  // just to make it invertable.
    invertable3x3cd_ut.at({0, 0}) = 2;  // just to make it invertable.
  }
  void TearDown() override {}
};

// ==================== shared linalg test helpers ====================
// Reusable helpers for the per-function linalg tests (isometry/SVD checks, fermionic operator
// builders, spectrum comparisons, and Krylov reference machinery).

// Helper: verify that X is an isometry / unitary over its auxiliary index `aux`: contracting the
// physical legs against X^dagger yields the identity on the aux index. Handles both conventions:
//   - aux is the FIRST leg (e.g. Svd vT):        checks X X^dagger = I
//   - aux is the LAST  leg (e.g. Svd/Eig U, V):  checks X^dagger X = I
// Per the fermionic contraction rule, the dagger operand is followed by fermion_twists() and placed
// on the left of the contraction.
inline void expect_unitary(const UniTensor &X, const std::string &aux, double tol) {
  UniTensor Xd = X.Dagger();
  Xd.relabel_(aux, aux + "_dag");
  UniTensor G = (X.labels().front() == aux) ? Contract(X.fermion_twists(), Xd)
                                            : Contract(Xd.fermion_twists(), X);
  UniTensor Id = 0. * G.clone();
  for (cytnx_uint64 k = 0; k < Id.shape()[0]; k++) {
    auto p = Id.at({k, k});
    p = 1.0;
  }
  EXPECT_TRUE((G.apply() - Id.apply()).Norm().item() < tol);
}

// Helper: verify an SVD decomposition usv = [S, U, vT]: U and vT are isometries (U^dagger U = I,
// vT vT^dagger = I) and the reconstruction A = U S vT holds. vT is the genuine
// right-singular-vector tensor (already living on the column space), so the reconstruction is the
// direct product -- no Dagger construction and no fermion_twists are needed (this is what
// distinguishes Svd/Gesvd from the eigen-decompositions, where only V is returned).
inline void expect_svd_reconstructs(const std::vector<UniTensor> &usv, const UniTensor &A,
                                    double tol) {
  ASSERT_EQ(usv.size(), 3);
  UniTensor S = usv[0], U = usv[1], vT = usv[2];
  expect_unitary(U, "_aux_L", tol);
  expect_unitary(vT, "_aux_R", tol);
  UniTensor recon = Contract(Contract(U, S), vT);
  recon.permute_(A.labels());
  EXPECT_TRUE((recon.apply() - A.apply()).Norm().item() < tol);
}

// Helper: square fermionic UniTensor (combined row space == combined column space)
// with one even (Qs 0) and one odd (Qs 1) sector, each of degeneracy 2.
inline UniTensor make_square_fermionic(const std::vector<std::string> &labels) {
  Bond Bi = Bond(BD_IN, {Qs(0) >> 2, Qs(1) >> 2}, {Symmetry::FermionParity()});
  Bond Bo = Bi.redirect();
  return UniTensor({Bi, Bo}, labels);
}

// Helper: rank-4 square fermionic UniTensor on row space (a,b), canonical leg order [a, b, a*, b*]
// (rowrank 2). Real-symmetric values -> Hermitian. A *consistent* permute (same permutation on
// row and column legs, e.g. {1,0,3,2}) then makes the pending signflip non-trivial while leaving
// the operator (hence its spectrum / singular values) unchanged.
inline UniTensor make_rank4_hermitian() {
  Bond a = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 1}, {Symmetry::FermionParity()});
  Bond b = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 1}, {Symmetry::FermionParity()});
  UniTensor M = UniTensor({a, b, a.redirect(), b.redirect()}, {"a", "b", "c", "d"});
  M.set_rowrank_(2);
  auto sh = M.shape();
  for (cytnx_uint64 i = 0; i < sh[0]; i++)
    for (cytnx_uint64 j = 0; j < sh[1]; j++)
      for (cytnx_uint64 k = 0; k < sh[2]; k++)
        for (cytnx_uint64 l = 0; l < sh[3]; l++) {
          auto p = M.at({i, j, k, l});
          if (!p.exists()) continue;
          cytnx_uint64 r = 2 * i + j, c = 2 * k + l;
          cytnx_uint64 lo = std::min(r, c), hi = std::max(r, c);
          p = 0.1 * double(1 + lo * 4 + hi) +
              0.2;  // depends only on the unordered {r,c} -> Hermitian
        }
  return M;
}

// Same operator at a chosen dtype: real-symmetric (Double) or genuinely complex-Hermitian
// (ComplexDouble). For the complex case the real part is symmetric and the imaginary part is
// antisymmetric (M[r,c] = conj(M[c,r])).
inline UniTensor make_rank4_herm_dt(bool cplx) {
  if (!cplx) return make_rank4_hermitian();
  Bond a = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 1}, {Symmetry::FermionParity()});
  Bond b = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 1}, {Symmetry::FermionParity()});
  UniTensor M =
    UniTensor({a, b, a.redirect(), b.redirect()}, {"a", "b", "c", "d"}, 2, Type.ComplexDouble);
  auto sh = M.shape();
  for (cytnx_uint64 i = 0; i < sh[0]; i++)
    for (cytnx_uint64 j = 0; j < sh[1]; j++)
      for (cytnx_uint64 k = 0; k < sh[2]; k++)
        for (cytnx_uint64 l = 0; l < sh[3]; l++) {
          auto p = M.at({i, j, k, l});
          if (!p.exists()) continue;
          cytnx_uint64 r = 2 * i + j, c = 2 * k + l;
          cytnx_uint64 lo = std::min(r, c), hi = std::max(r, c);
          double re = 0.1 * double(1 + lo * 4 + hi) + 0.2;
          double im = (r == c) ? 0.0 : (r < c ? 1.0 : -1.0) * (0.05 * double(1 + lo * 4 + hi));
          p = cytnx_complex128(re, im);
        }
  return M;
}

// Helper: build a fermionic UniTensor with mixed in/out legs on both row and column spaces and
// degeneracies (rowrank 2), filled with sequential values over its existing components.
inline UniTensor make_mixed_inout_fermionic() {
  Bond B5Li = Bond(BD_IN, {Qs(0), Qs(1)}, {2, 1}, {Symmetry::FermionParity()});
  Bond B5Lo = Bond(BD_OUT, {Qs(0), Qs(1)}, {1, 2}, {Symmetry::FermionParity()});
  Bond B5Ri = Bond(BD_IN, {Qs(0), Qs(1)}, {1, 2}, {Symmetry::FermionParity()});
  Bond B5Ro = Bond(BD_OUT, {Qs(0), Qs(1)}, {2, 1}, {Symmetry::FermionParity()});
  UniTensor M = UniTensor({B5Li, B5Lo, B5Ri, B5Ro}, {"li", "lo", "ri", "ro"});
  M.set_rowrank_(2);
  cytnx_double val = 1.0;
  auto sh = M.shape();
  for (cytnx_uint64 i = 0; i < sh[0]; i++)
    for (cytnx_uint64 j = 0; j < sh[1]; j++)
      for (cytnx_uint64 k = 0; k < sh[2]; k++)
        for (cytnx_uint64 l = 0; l < sh[3]; l++) {
          auto proxy = M.at({i, j, k, l});
          if (proxy.exists()) {
            proxy = val;
            val += 1.0;
          }
        }
  return M;
}

// Helper: collect the diagonal entries (real parts) of a (block-)diagonal UniTensor, sorted
// ascending. Eigenvalues / singular values are invariant under a consistent permute, so comparing
// these between a tensor and its permuted (sign-flip-active) form verifies correct sign handling.
inline std::vector<double> sorted_diagonal(UniTensor d) {
  std::vector<double> vals;
  auto sh = d.shape();
  for (cytnx_uint64 k = 0; k < sh[0]; k++) {
    auto p = d.at({k, k});
    if (p.exists()) vals.push_back(double(p.real()));
  }
  std::sort(vals.begin(), vals.end());
  return vals;
}

// Helper: assert two (block-)diagonal tensors carry the same (real) diagonal multiset. Used to
// cross-check spectra/singular values across solvers (Eigh vs Eig, Gesvd vs Svd) and across a
// consistent sign-flip permute.
inline void expect_same_diagonal(const UniTensor &a, const UniTensor &b, double tol) {
  auto va = sorted_diagonal(a), vb = sorted_diagonal(b);
  ASSERT_EQ(va.size(), vb.size());
  for (size_t i = 0; i < va.size(); i++) EXPECT_NEAR(va[i], vb[i], tol);
}

// Helper: build a permuted (consistent {1,0,3,2}) copy and assert it carries non-trivial sign
// flips.
inline UniTensor permute_with_signflips(const UniTensor &M) {
  UniTensor Mp = M.permute({1, 0, 3, 2}).contiguous();
  bool anyflip = false;
  for (auto f : Mp.signflip()) anyflip = anyflip || f;
  EXPECT_TRUE(anyflip);  // ensure the signflip negation path is actually exercised
  return Mp;
}

// Helper: given the (diagonal) eigenvalue tensors of exp(a*M) and of M, assert their sorted real
// diagonals satisfy exp_evals[i] == exp(a * evals[i]). exp is monotonic for real a, so the
// ascending orders match.
inline void expect_exp_spectrum(const UniTensor &exp_evals, const UniTensor &evals, double a,
                                double tol) {
  auto se = sorted_diagonal(exp_evals);
  auto sm = sorted_diagonal(evals);
  ASSERT_EQ(se.size(), sm.size());
  for (size_t i = 0; i < se.size(); i++) EXPECT_NEAR(se[i], std::exp(a * sm[i]), tol);
}

// Helper: assert two eigenvector tensors (physical legs + aux as the LAST leg) describe the same
// eigenvectors up to a per-vector phase. For orthonormal eigenbases Ua^dagger Ub is then a diagonal
// unitary: each diagonal entry is a unit-modulus phase and the off-diagonals vanish. Assumes a
// non-degenerate spectrum with matching sorted ordering in both inputs (true for Eigh of M and of a
// monotonic function of M). Per the fermionic contraction rule the dagger operand is followed by
// fermion_twists().
inline void expect_same_eigenvectors(const UniTensor &Ua, const UniTensor &Ub,
                                     const std::string &aux, double tol) {
  UniTensor Uad = Ua.Dagger();
  Uad.relabel_(aux, aux + "_a");
  UniTensor G = Contract(Uad.fermion_twists(), Ub).apply();  // -> [aux_a, aux]
  for (cytnx_uint64 k = 0; k < G.shape()[0]; k++) {
    auto p = G.at({k, k});
    if (p.exists()) {
      EXPECT_NEAR(double(Scalar(p).abs()), 1.0, tol);  // diagonal is a unit-modulus phase
      p = 0.;  // remove it; only the (vanishing) off-diagonals should remain
    }
  }
  EXPECT_TRUE(G.Norm().item() < tol);
}

// Fermionic Krylov test operator Op = A^dag A, built from a 4-leg fermionic tensor A.
//
// Creates A as a 4-leg tensor created in a *permuted* leg order with some sign flips.
// Forms the linear, Hermitian operator A^dag A. Its spectrum equals the squared singular values of
// A. The initial Krylov vector is likewise created permuted, and carries a non-trivial signflip;

// number of stored elements in a (symmetric) ket = sum of block sizes (the LinOp dimension nx).
inline cytnx_uint64 ferm_ket_nx(const UniTensor &v) {
  cytnx_uint64 n = 0;
  for (const auto &blk : v.get_blocks_()) {
    cytnx_uint64 d = 1;
    for (auto s : blk.shape()) d *= (cytnx_uint64)s;
    n += d;
  }
  return n;
}

// 4-leg A [x,y,c,d] (rowrank 2), each leg with two even + two odd states, filled as
// identity + small perturbation (so A^dag A is well-conditioned and non-degenerate -- a clean,
// distinct ground state for the Krylov solvers), then permuted {1,0,3,2} so the pending signflip
// is non-trivial (several blocks with, several without). The charge-0 column sector has dimension
// 8 (large enough for ARPACK's ncv constraint 2+k <= ncv <= nx).
inline UniTensor make_ferm_A() {
  Bond x = Bond(BD_IN, {Qs(0) >> 2, Qs(1) >> 2}, {Symmetry::FermionParity()});
  Bond y = Bond(BD_IN, {Qs(0) >> 2, Qs(1) >> 2}, {Symmetry::FermionParity()});
  UniTensor A = UniTensor({x, y, x.redirect(), y.redirect()}, {"x", "y", "c", "d"});
  A.set_rowrank_(2);
  auto sh = A.shape();
  for (cytnx_uint64 i = 0; i < sh[0]; i++)
    for (cytnx_uint64 j = 0; j < sh[1]; j++)
      for (cytnx_uint64 k = 0; k < sh[2]; k++)
        for (cytnx_uint64 l = 0; l < sh[3]; l++) {
          auto p = A.at({i, j, k, l});
          if (p.exists()) {
            double diag = (i == k && j == l) ? 1.0 : 0.0;
            double pert = 0.13 * (double(((i + 1) * (k + 2) + (j + 1) * (l + 3)) % 7) * 0.1);
            p = diag + pert;
          }
        }
  UniTensor Ap = A.permute({1, 0, 3, 2}).contiguous();
  bool anyflip = false;
  for (auto f : Ap.signflip()) anyflip = anyflip || f;
  EXPECT_TRUE(anyflip);  // operator factor carries non-trivial fermionic signs
  return Ap;
}

// Op psi = A^dag (A psi); psi lives in A's column space.
inline UniTensor ferm_ada_apply(const UniTensor &A, const UniTensor &psi) {
  UniTensor phi = Contract(A.fermion_twists(), psi);  // -> A's row space
  return Contract(A.Dagger().fermion_twists(), phi);  // -> A's column space, psi's order
}

class FermiAdaOp : public LinOp {
 public:
  UniTensor A;
  FermiAdaOp(const UniTensor &A_, const cytnx_uint64 &nx)
      : LinOp("mv", nx, A_.dtype(), A_.device()), A(A_) {}
  UniTensor matvec(const UniTensor &psi) override { return ferm_ada_apply(A, psi); }
};

// Creates a ket in A's column space with no signflips, in the leg order that
// ferm_ada_apply produces (A's column labels reversed), so the matvec needs no reorder.
inline UniTensor make_clean_ket(const UniTensor &A) {
  UniTensor v =
    UniTensor({A.bonds()[3].redirect(), A.bonds()[2].redirect()}, {A.labels()[3], A.labels()[2]});
  v.set_rowrank_(2);
  return v;
}

// the discriminating initial vector: the same signflip-free ket but *created in reversed leg order
// and permuted into A's column order*, so it creates a non-trivial signflip in the contraction
inline UniTensor make_ferm_ada_ket(const UniTensor &A) {
  // build in the opposite leg order, then permute into the matvec's output order (==
  // make_clean_ket order); the permute is what injects the pending signflip.
  UniTensor vr =
    UniTensor({A.bonds()[2].redirect(), A.bonds()[3].redirect()}, {A.labels()[2], A.labels()[3]});
  vr.set_rowrank_(2);
  auto sh = vr.shape();
  double val = 1.0;
  for (cytnx_uint64 a = 0; a < sh[0]; a++)
    for (cytnx_uint64 b = 0; b < sh[1]; b++) {
      auto p = vr.at({a, b});
      if (p.exists()) {  // fill every charge-0 component -> overlaps every eigenvector
        p = 0.2 + 0.1 * val;
        val += 1.0;
      }
    }
  UniTensor v = vr.permute(std::vector<cytnx_int64>{1, 0})
                  .contiguous();  // -> A's column order, carries pending signflip
  bool anyflip = false;
  for (auto f : v.signflip()) anyflip = anyflip || f;
  EXPECT_TRUE(anyflip);  // initial vector is sign-flip-active
  return v;
}

// signed fermionic inner product <P|Q> (fully contracted -> a Dense scalar, so .item() is valid).
inline double ferm_fdot_real(const UniTensor &P, const UniTensor &Q) {
  return double(Contract(P.Dagger().fermion_twists(), Q).item().real());
}

// eigenvector fidelity |<v|w>| / sqrt(<v|v><w|w>): phase-invariant, ~1 for the same eigenvector.
// |<v|w>| is the Norm() (a Dense scalar magnitude) of the fermionic overlap, so it is valid for
// complex (Arnoldi) eigenvectors too; Norm() is signflip-independent so no apply() is needed.
inline double ferm_fidelity(const UniTensor &v, const UniTensor &w) {
  double ip = double(Contract(v.Dagger().fermion_twists(), w).Norm().item().real());
  return ip / (double(v.Norm().item().real()) * double(w.Norm().item().real()));
}

// squared singular values of A = the spectrum of Op = A^dag A (independent sign-correct
// reference).
inline std::vector<double> ferm_sigma2(const UniTensor &A) {
  UniTensor S = linalg::Svd(A, false)[0];
  std::vector<double> out;
  for (auto &bk : S.get_blocks_()) {
    Storage st = bk.clone().contiguous().storage();
    for (cytnx_uint64 i = 0; i < st.size(); i++) {
      double s = st.at<cytnx_double>(i);
      out.push_back(s * s);
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

// the kmax lowest eigenpairs of Op = A^dag A within the charge-0 ket sector, by explicit dense
// diagonalization (independent of the Krylov solvers and of the library's sign-frame pinning -- it
// acts on clean, signflip-resolved basis kets). Returns {(eigenvalue, eigenvector ket)} sorted
// ascending by eigenvalue.
inline std::vector<std::pair<double, UniTensor>> ferm_dense_lowest(const UniTensor &A,
                                                                   cytnx_uint64 kmax) {
  UniTensor proto = make_clean_ket(A);
  auto sh = proto.shape();
  std::vector<std::vector<cytnx_uint64>> comps;
  for (cytnx_uint64 a = 0; a < sh[0]; a++)
    for (cytnx_uint64 b = 0; b < sh[1]; b++)
      if (proto.at({a, b}).exists()) comps.push_back({a, b});
  cytnx_uint64 n = comps.size();
  std::vector<UniTensor> basis(n);
  for (cytnx_uint64 i = 0; i < n; i++) {
    UniTensor e = proto.clone();
    e.at(comps[i]) = 1.0;
    basis[i] = e;
  }
  Tensor M = zeros({(cytnx_int64)n, (cytnx_int64)n});
  for (cytnx_uint64 c = 0; c < n; c++) {
    UniTensor w = ferm_ada_apply(A, basis[c]);
    for (cytnx_uint64 j = 0; j < n; j++) M.at({j, c}) = ferm_fdot_real(basis[j], w);
  }
  auto eg = linalg::Eigh(M);  // [eigvals(n), eigvecs(n x n, columns)]
  Storage ev = eg[0].storage(), evec = eg[1].storage();
  std::vector<cytnx_uint64> idx(n);
  for (cytnx_uint64 i = 0; i < n; i++) idx[i] = i;
  std::sort(idx.begin(), idx.end(), [&](cytnx_uint64 a, cytnx_uint64 b) {
    return ev.at<cytnx_double>(a) < ev.at<cytnx_double>(b);
  });
  std::vector<std::pair<double, UniTensor>> out;
  for (cytnx_uint64 r = 0; r < kmax && r < n; r++) {
    cytnx_uint64 g = idx[r];
    UniTensor w = proto.clone();
    w *= 0.0;
    for (cytnx_uint64 j = 0; j < n; j++) {
      UniTensor t = basis[j].clone();
      t *= evec.at<cytnx_double>(j * n + g);  // column g, row j (eigenvectors are columns)
      w += t;
    }
    out.push_back({ev.at<cytnx_double>(g), w});
  }
  return out;
}
// assert E is a genuine eigenvalue of Op = A^dag A (E in sigma(A)^2) and v is an eigenvector
// (||Op v - E v|| ~ 0), using the fermion-correct matvec -> validates the sign handling. A is
// promoted to v's dtype here (not inside ferm_ada_apply): the real solve never needs it, only this
// post-hoc residual on Arnoldi's complex eigenvector does.
inline void expect_ada_eigenpair(const UniTensor &A, double E, const UniTensor &v, double tol) {
  UniTensor Ov = ferm_ada_apply(A.astype(v.dtype()), v);
  UniTensor Ev = E * v;
  Ev.apply_();
  EXPECT_TRUE((Ov - Ev).Norm().item() < tol);  // v is an eigenvector with eigenvalue E
  auto s2 = ferm_sigma2(A);
  double best = 1e18;
  for (double s : s2) best = std::min(best, std::abs(E - s));
  EXPECT_TRUE(best < tol);  // E matches a squared singular value of A (sign-correct reference)
}

// Validate a Krylov result for the k lowest states against the dense reference `low` (ascending).
// eigs[0] holds the k eigenvalues; eigs[1..k] are the eigenvectors. For each state: eigenvalue
// matches (and is genuine), eigenvector is normalized and matches the dense eigenvector
// (|<v|w>| ~ 1); and distinct eigenvectors are mutually orthogonal (|<v_i|v_j>| ~ 0).
inline void expect_lowest_states(const UniTensor &A, const std::vector<UniTensor> &eigs,
                                 const std::vector<std::pair<double, UniTensor>> &low, double tol) {
  cytnx_uint64 k = low.size();
  UniTensor evals = eigs[0];  // non-const handle so .at(...) returns a mutable proxy
  for (cytnx_uint64 i = 0; i < k; i++) {
    double E = double(evals.at({i}).real());
    EXPECT_NEAR(E, low[i].first, tol);  // eigenvalue matches dense (both ascending)
    expect_ada_eigenpair(A, E, eigs[i + 1], tol);  // residual + sigma^2 membership
    EXPECT_NEAR(ferm_fidelity(eigs[i + 1], low[i].second), 1.0, 1e-5);  // per-state fidelity
    // normalized: Norm() (Frobenius) is signflip-independent, so no apply() is needed
    EXPECT_NEAR(double(eigs[i + 1].Norm().item().real()), 1.0, 1e-6);
  }
  for (cytnx_uint64 i = 0; i < k; i++)
    for (cytnx_uint64 j = i + 1; j < k; j++)
      EXPECT_LT(ferm_fidelity(eigs[i + 1], eigs[j + 1]), 1e-5);  // orthogonal eigenvectors
}

#endif
