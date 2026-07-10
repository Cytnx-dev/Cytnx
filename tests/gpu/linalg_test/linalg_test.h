#ifndef _H_linalg_test
#define _H_linalg_test

#include "cytnx.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

using namespace cytnx;

class linalg_Test : public ::testing::Test {
 public:
  // ==================== general ===================
  Tensor arange3x3d = arange(0, 9, 1, Type.Double).reshape(3, 3).to(cytnx::Device.cuda);
  Tensor ones3x3d = ones(9, Type.Double).reshape(3, 3).to(cytnx::Device.cuda);
  Tensor eye3x3d = eye(3, Type.Double).to(cytnx::Device.cuda);
  Tensor zeros3x3d = zeros(9, Type.Double).reshape(3, 3).to(cytnx::Device.cuda);

  Tensor arange3x3cd = arange(0, 9, 1, Type.ComplexDouble).reshape(3, 3).to(cytnx::Device.cuda) +
                       cytnx_complex128(0, 1) *
                         arange(0, 9, 1, Type.ComplexDouble).reshape(3, 3).to(cytnx::Device.cuda);
  Tensor ones3x3cd = ones(9, Type.ComplexDouble).reshape(3, 3).to(cytnx::Device.cuda);
  Tensor eye3x3cd = eye(3, Type.ComplexDouble).to(cytnx::Device.cuda);
  Tensor zeros3x3cd = zeros(9, Type.ComplexDouble).reshape(3, 3).to(cytnx::Device.cuda);

  std::string data_dir = CYTNX_TEST_DATA_DIR "/linalg/";
  // ==================== svd_truncate ===================
  Bond svd_I = Bond(BD_OUT, {Qs(1), Qs(-1)}, {1, 1});
  Bond svd_J =
    Bond(BD_IN, {Qs(5), Qs(3), Qs(1), Qs(-1), Qs(-3), Qs(-5), Qs(-7)}, {6, 22, 57, 68, 38, 8, 1});
  Bond svd_K =
    Bond(BD_OUT, {Qs(5), Qs(3), Qs(1), Qs(-1), Qs(-3), Qs(-5), Qs(-7)}, {6, 22, 57, 68, 38, 8, 1});
  Bond svd_L = Bond(BD_OUT, {Qs(1), Qs(-1)}, {1, 1});
  UniTensor svd_T = UniTensor({svd_I, svd_J, svd_K, svd_L}, {"a", "b", "c", "d"}, 1, Type.Double,
                              Device.cuda, false)
                      .to(cytnx::Device.cuda);

  UniTensor svd_T_dense =
    UniTensor(arange(0, 11 * 13, 1).reshape(11, 13)).astype(Type.ComplexDouble).to(Device.cuda);

  Tensor svd_Sans;
  //==================== Lanczos_Gnd_Ut ===================
  Tensor A = Tensor::Load(data_dir + "Lanczos_Gnd/lan_block_A.cytn").to(cytnx::Device.cuda);
  Tensor B = Tensor::Load(data_dir + "Lanczos_Gnd/lan_block_B.cytn").to(cytnx::Device.cuda);
  Tensor C = Tensor::Load(data_dir + "Lanczos_Gnd/lan_block_C.cytn").to(cytnx::Device.cuda);
  Bond lan_I = Bond(BD_IN, {Qs(-1), Qs(0), Qs(1)}, {9, 9, 9});
  Bond lan_J = Bond(BD_OUT, {Qs(-1), Qs(0), Qs(1)}, {9, 9, 9});
  UniTensor H = UniTensor({lan_I, lan_J}).to(cytnx::Device.cuda);

  //==================== QR ===================
  Tensor Qr_Qans = Tensor::Load(data_dir + "Qr/qr_Qans.cytn").to(cytnx::Device.cuda);
  Tensor Qr_Rans = Tensor::Load(data_dir + "Qr/qr_Rans.cytn").to(cytnx::Device.cuda);

  //==================== ExpH ===================
  Tensor expH_ans = Tensor::Load(data_dir + "expH/expH_ans.cytn").to(cytnx::Device.cuda);

  //==================== Pow ===================
  Tensor Pow_ans = Tensor::Load(data_dir + "Pow/Pow_ans.cytn").to(cytnx::Device.cuda);

  //==================== Mod ===================
  Tensor Mod_ans = Tensor::Load(data_dir + "Mod/Mod_ans.cytn").to(cytnx::Device.cuda);
  Tensor ModUtUt_ans = Tensor::Load(data_dir + "Mod/ModUtUt_ans.cytn").to(cytnx::Device.cuda);

 protected:
  void SetUp() override {
    //================ svd truncate =======================

    svd_T = svd_T.Load(data_dir + "Svd_truncate/Svd_truncate1.cytnx").to(cytnx::Device.cuda);
    svd_T.permute_({1, 0, 3, 2});
    svd_T.contiguous_();
    svd_T.set_rowrank_(2);
    svd_Sans = Tensor::Load(data_dir + "Svd_truncate/S_truncate1.cytn").to(cytnx::Device.cuda);
    svd_Sans = algo::Sort(svd_Sans);
    //==================== Lanczos_Gnd_Ut ===================
    H.put_block(A, 0);
    H.put_block(B, 1);
    H.put_block(C, 2);
    H.relabel_({"a", "b"});
  }
  void TearDown() override {}
};

// ==================== shared linalg test helpers ====================
// Reusable helpers for the per-function GPU linalg tests: isometry/SVD reconstruction checks,
// fermionic operator builders, spectrum comparisons, and Krylov reference machinery.

inline std::vector<double> sorted_diagonal(UniTensor d) {
  std::vector<double> v;
  auto sh = d.shape();
  for (cytnx_uint64 k = 0; k < sh[0]; k++) {
    auto p = d.at({k, k});
    if (p.exists()) v.push_back(double(p.real()));
  }
  std::sort(v.begin(), v.end());
  return v;
}

inline void expect_same(const std::vector<double> &a, const std::vector<double> &b, double tol) {
  ASSERT_EQ(a.size(), b.size());
  for (size_t i = 0; i < a.size(); i++) EXPECT_NEAR(a[i], b[i], tol);
}

// X is an isometry over its auxiliary index `aux`: X^dagger X = I (aux last) or X X^dagger = I
// (aux first). The dagger operand is followed by fermion_twists() per the fermionic contraction.
inline void expect_unitary(const UniTensor &X, const std::string &aux, double tol) {
  UniTensor Xd = X.Dagger();
  Xd.relabel_(aux, aux + "_dag");
  UniTensor G = (X.labels().front() == aux) ? Contract(X.fermion_twists(), Xd)
                                            : Contract(Xd.fermion_twists(), X);
  UniTensor Id = 0. * G.clone();
  for (cytnx_uint64 k = 0; k < Id.shape()[0]; k++) Id.at({k, k}) = 1.0;
  EXPECT_TRUE((G.apply() - Id.apply()).Norm().item() < tol);
}

// SVD outputs [S,U,vT] are self-consistent: U, vT are isometries and U S vT == A.
inline void expect_svd_reconstructs(const std::vector<UniTensor> &usv, const UniTensor &A,
                                    double tol) {
  ASSERT_EQ(usv.size(), 3);
  expect_unitary(usv[1], "_aux_L", tol);
  expect_unitary(usv[2], "_aux_R", tol);
  UniTensor recon = Contract(Contract(usv[1], usv[0]), usv[2]).permute(A.labels());
  EXPECT_TRUE((recon.apply() - A.apply()).Norm().item() < tol);
}

// rank-4 [a,b,a*,b*] Hermitian fermionic operator (real-symmetric per sector).
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
          p = 0.1 * double(1 + lo * 4 + hi) + 0.2;
        }
  return M;
}

// consistent {1,0,3,2} permute makes the pending signflip non-trivial (sign-flip-active).
inline UniTensor permute_with_signflips(const UniTensor &M) {
  UniTensor Mp = M.permute({1, 0, 3, 2}).contiguous();
  bool anyflip = false;
  for (auto f : Mp.signflip()) anyflip = anyflip || f;
  EXPECT_TRUE(anyflip);
  return Mp;
}

inline cytnx_uint64 ferm_ket_nx(const UniTensor &v) {
  cytnx_uint64 n = 0;
  for (const auto &blk : v.get_blocks_()) {
    cytnx_uint64 d = 1;
    for (auto s : blk.shape()) d *= (cytnx_uint64)s;
    n += d;
  }
  return n;
}

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
  EXPECT_TRUE(anyflip);
  return Ap;
}

// Op psi = A^dag (A psi). Returns the raw contraction result: the solver's matvec wrapper resolves
// the signflip frame and makes it contiguous, and psi is built (below) in this contraction's leg
// order so no permute is needed (a permute would swap fermionic legs and wrongly negate the
// odd-odd sector).
inline UniTensor ferm_ada_apply(const UniTensor &A, const UniTensor &psi) {
  UniTensor phi = Contract(A.fermion_twists(), psi);
  return Contract(A.Dagger().fermion_twists(), phi);
}

class FermiAdaOp : public LinOp {
 public:
  UniTensor A;
  FermiAdaOp(const UniTensor &A_, const cytnx_uint64 &nx)
      : LinOp("mv", nx, A_.dtype(), A_.device()), A(A_) {}
  UniTensor matvec(const UniTensor &psi) override { return ferm_ada_apply(A, psi); }
};

inline UniTensor make_clean_ket(const UniTensor &A) {
  UniTensor v =
    UniTensor({A.bonds()[3].redirect(), A.bonds()[2].redirect()}, {A.labels()[3], A.labels()[2]});
  v.set_rowrank_(2);
  return v;
}

// sign-flip-active initial vector (built in the opposite leg order then permuted into the matvec's
// output order, which injects the pending signflip).
inline UniTensor make_ferm_ada_ket(const UniTensor &A) {
  UniTensor vr =
    UniTensor({A.bonds()[2].redirect(), A.bonds()[3].redirect()}, {A.labels()[2], A.labels()[3]});
  vr.set_rowrank_(2);
  auto sh = vr.shape();
  double val = 1.0;
  for (cytnx_uint64 a = 0; a < sh[0]; a++)
    for (cytnx_uint64 b = 0; b < sh[1]; b++) {
      auto p = vr.at({a, b});
      if (p.exists()) {
        p = 0.2 + 0.1 * val;
        val += 1.0;
      }
    }
  UniTensor v = vr.permute(std::vector<cytnx_int64>{1, 0}).contiguous();
  bool anyflip = false;
  for (auto f : v.signflip()) anyflip = anyflip || f;
  EXPECT_TRUE(anyflip);
  return v;
}

inline double ferm_fdot_real(const UniTensor &P, const UniTensor &Q) {
  return double(Contract(P.Dagger().fermion_twists(), Q).item().real());
}
// |<v|w>| / sqrt(<v|v><w|w>): phase-invariant. |<v|w>| is the magnitude (Norm()) of the fermionic
// overlap (valid for complex eigenvectors); Norm() is signflip-independent so no apply() needed.
inline double ferm_fidelity(const UniTensor &v, const UniTensor &w) {
  double ip = double(Contract(v.Dagger().fermion_twists(), w).Norm().item().real());
  return ip / (double(v.Norm().item().real()) * double(w.Norm().item().real()));
}

// kmax lowest eigenpairs of O = A^dag A in the charge-0 ket sector by explicit dense Eigh (CPU).
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
  auto eg = linalg::Eigh(M);
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
      t *= evec.at<cytnx_double>(j * n + g);
      w += t;
    }
    out.push_back({ev.at<cytnx_double>(g), w});
  }
  return out;
}

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

// eigs[] are brought back to the CPU before this check (A and the dense reference are on CPU).
inline void expect_lowest_states(const UniTensor &A, const std::vector<UniTensor> &eigs,
                                 const std::vector<std::pair<double, UniTensor>> &low, double tol) {
  cytnx_uint64 k = low.size();
  auto s2 = ferm_sigma2(A);
  UniTensor evals = eigs[0];  // non-const handle so .at(...) returns a mutable proxy
  for (cytnx_uint64 i = 0; i < k; i++) {
    double E = double(evals.at({i}).real());
    EXPECT_NEAR(E, low[i].first, tol);
    // A is promoted to the eigenvector dtype here (only Arnoldi's complex eigenvector needs it).
    UniTensor Ov = ferm_ada_apply(A.astype(eigs[i + 1].dtype()), eigs[i + 1]);
    UniTensor Ev = E * eigs[i + 1];
    Ev.apply_();
    EXPECT_TRUE((Ov - Ev).Norm().item() < tol);  // residual
    double best = 1e18;
    for (double s : s2) best = std::min(best, std::abs(E - s));
    EXPECT_TRUE(best < tol);  // E in sigma(A)^2
    EXPECT_NEAR(ferm_fidelity(eigs[i + 1], low[i].second), 1.0, 1e-5);
    // normalized: Norm() (Frobenius) is signflip-independent, so no apply() is needed
    EXPECT_NEAR(double(eigs[i + 1].Norm().item().real()), 1.0, 1e-6);
  }
  for (cytnx_uint64 i = 0; i < k; i++)
    for (cytnx_uint64 j = i + 1; j < k; j++)
      EXPECT_LT(ferm_fidelity(eigs[i + 1], eigs[j + 1]), 1e-5);
}

inline std::vector<UniTensor> to_cpu(std::vector<UniTensor> eigs) {
  for (auto &e : eigs) e = e.to(Device.cpu);
  return eigs;
}

#endif
