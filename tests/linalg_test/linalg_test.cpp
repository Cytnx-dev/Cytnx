#include "linalg_test.h"
#include "test_tools.h"

#include <complex>

using namespace cytnx;
using namespace testing;
using namespace TestTools;

namespace {
  Tensor SortedBlockSingularValues(const UniTensor &S) {
    Tensor all_svals = S.get_block_(0);
    for (cytnx_int64 i = 1; i < S.Nblocks(); i++) {
      all_svals = algo::Concatenate(all_svals, S.get_block_(i));
    }
    return algo::Sort(all_svals);
  }
}  // namespace

TEST(linalgKronTest, PadsLowerRankLhsOnLeft) {
  Tensor lhs = zeros({2}, Type.Double);
  lhs.at<double>({0}) = 10.0;
  lhs.at<double>({1}) = 20.0;
  Tensor rhs = arange(1, 13, 1, Type.Double).reshape(3, 4);

  Tensor out = linalg::Kron(lhs, rhs, true, false);

  EXPECT_EQ(out.shape(), (std::vector<cytnx_uint64>{3, 8}));
  for (cytnx_uint64 i = 0; i < out.shape()[0]; i++) {
    for (cytnx_uint64 j = 0; j < out.shape()[1]; j++) {
      const double expected =
        lhs.at<double>({j / rhs.shape()[1]}) * rhs.at<double>({i, j % rhs.shape()[1]});
      EXPECT_DOUBLE_EQ(out.at<double>({i, j}), expected);
    }
  }
}

TEST(linalgTensordotTest, RejectsRankZeroAxis) {
  Tensor scalar({}, Type.Double);

  EXPECT_THROW({ linalg::Tensordot(scalar, scalar, {0}, {0}); }, std::logic_error);
}

TEST(linalgTensordotTest, RejectsDiagRankZeroAxis) {
  Tensor diag = zeros({2}, Type.Double);
  Tensor scalar({}, Type.Double);

  EXPECT_THROW({ linalg::Tensordot_dg(diag, scalar, {0}, {0}, true); }, std::logic_error);
  EXPECT_THROW({ linalg::Tensordot_dg(scalar, diag, {0}, {0}, false); }, std::logic_error);
}

TEST_F(linalg_Test, BkUt_Svd_truncate1) {
  std::vector<UniTensor> res = linalg::Svd_truncate(svd_T, 200, 0, true);
  std::vector<double> vnm_S;
  for (size_t i = 0; i < res[0].shape()[0]; i++)
    vnm_S.push_back((double)(res[0].at({i, i}).real()));
  std::sort(vnm_S.begin(), vnm_S.end());
  for (size_t i = 0; i < vnm_S.size(); i++)
    EXPECT_TRUE(abs(vnm_S[i] - (double)(svd_Sans.at({0, i}).real())) < 1e-5);
  auto con_T1 = Contract(Contract(res[2], res[0]), res[1]);
  auto con_T2 = Contract(Contract(res[1], res[0]), res[2]);
}

TEST_F(linalg_Test, BkUt_Svd_truncate2) {
  std::vector<UniTensor> res = linalg::Svd_truncate(svd_T, 200, 1e-1, true);
  std::vector<double> vnm_S;
  for (size_t i = 0; i < res[0].shape()[0]; i++)
    vnm_S.push_back((double)(res[0].at({i, i}).real()));
  std::sort(vnm_S.begin(), vnm_S.end());
  for (size_t i = 0; i < vnm_S.size(); i++) EXPECT_TRUE(vnm_S[i] > 1e-1);
  auto con_T1 = Contract(Contract(res[2], res[0]), res[1]);
  auto con_T2 = Contract(Contract(res[1], res[0]), res[2]);
}

TEST_F(linalg_Test, BkUt_Svd_truncate_return_err_returns_discarded_values) {
  std::vector<UniTensor> full = linalg::Svd_truncate(svd_T, 999, 0, true, 0);
  std::vector<UniTensor> trunc = linalg::Svd_truncate(svd_T, 5, 0, true, 999);
  Tensor all_svals = SortedBlockSingularValues(full[0]);

  ASSERT_EQ(full.size(), 3);
  ASSERT_EQ(trunc.size(), 4);
  ASSERT_EQ(all_svals.shape()[0], 400);
  // keepdim may only be exceeded when singular values at the cut are exactly degenerate
  // (see the degeneracy note on Svd_truncate): every extra kept value must equal the
  // smallest value required by keepdim.
  cytnx_uint64 kept = trunc[0].shape()[0];
  ASSERT_GE(kept, 5);
  ASSERT_LE(kept, all_svals.shape()[0]);
  for (cytnx_uint64 j = all_svals.shape()[0] - kept; j < all_svals.shape()[0] - 5; j++) {
    EXPECT_EQ(all_svals.at({j}), all_svals.at({all_svals.shape()[0] - 5}));
  }
  ASSERT_EQ(trunc[3].shape()[0], all_svals.shape()[0] - kept);

  for (cytnx_uint64 i = 0; i < trunc[3].shape()[0]; i++) {
    EXPECT_EQ(all_svals.at({trunc[3].shape()[0] - 1 - i}), trunc[3].at({i}));
  }
}

TEST_F(linalg_Test, BkUt_Gesvd_truncate_return_err_returns_discarded_values) {
  std::vector<UniTensor> full = linalg::Gesvd_truncate(svd_T, 999, 0, true, true, 0);
  std::vector<UniTensor> trunc = linalg::Gesvd_truncate(svd_T, 5, 0, true, true, 999);
  Tensor all_svals = SortedBlockSingularValues(full[0]);

  ASSERT_EQ(full.size(), 3);
  ASSERT_EQ(trunc.size(), 4);
  ASSERT_EQ(all_svals.shape()[0], 400);
  // keepdim may only be exceeded when singular values at the cut are exactly degenerate
  // (see the degeneracy note on Svd_truncate): every extra kept value must equal the
  // smallest value required by keepdim.
  cytnx_uint64 kept = trunc[0].shape()[0];
  ASSERT_GE(kept, 5);
  ASSERT_LE(kept, all_svals.shape()[0]);
  for (cytnx_uint64 j = all_svals.shape()[0] - kept; j < all_svals.shape()[0] - 5; j++) {
    EXPECT_EQ(all_svals.at({j}), all_svals.at({all_svals.shape()[0] - 5}));
  }
  ASSERT_EQ(trunc[3].shape()[0], all_svals.shape()[0] - kept);

  for (cytnx_uint64 i = 0; i < trunc[3].shape()[0]; i++) {
    EXPECT_EQ(all_svals.at({trunc[3].shape()[0] - 1 - i}), trunc[3].at({i}));
  }
}

TEST_F(linalg_Test, BkUt_Svd_truncate_return_err_one_returns_first_discarded_value) {
  std::vector<UniTensor> full = linalg::Svd_truncate(svd_T, 999, 0, true, 0);
  std::vector<UniTensor> trunc = linalg::Svd_truncate(svd_T, 5, 0, true, 1);
  Tensor all_svals = SortedBlockSingularValues(full[0]);

  ASSERT_EQ(full.size(), 3);
  ASSERT_EQ(trunc.size(), 4);
  ASSERT_TRUE(trunc[3].shape().empty());
  // the kept dimension adapts to exact degeneracies at the cut (>= keepdim); the returned
  // error is the largest singular value that was actually discarded.
  cytnx_uint64 kept = trunc[0].shape()[0];
  ASSERT_GE(kept, 5);
  ASSERT_LT(kept, all_svals.shape()[0]);
  EXPECT_EQ(all_svals.at({all_svals.shape()[0] - kept - 1}), trunc[3].at({}));
}

TEST_F(linalg_Test, BkUt_Gesvd_truncate_return_err_one_returns_first_discarded_value) {
  std::vector<UniTensor> full = linalg::Gesvd_truncate(svd_T, 999, 0, true, true, 0);
  std::vector<UniTensor> trunc = linalg::Gesvd_truncate(svd_T, 5, 0, true, true, 1);
  Tensor all_svals = SortedBlockSingularValues(full[0]);

  ASSERT_EQ(full.size(), 3);
  ASSERT_EQ(trunc.size(), 4);
  ASSERT_TRUE(trunc[3].shape().empty());
  // the kept dimension adapts to exact degeneracies at the cut (>= keepdim); the returned
  // error is the largest singular value that was actually discarded.
  cytnx_uint64 kept = trunc[0].shape()[0];
  ASSERT_GE(kept, 5);
  ASSERT_LT(kept, all_svals.shape()[0]);
  EXPECT_EQ(all_svals.at({all_svals.shape()[0] - kept - 1}), trunc[3].at({}));
}

/*=====test info=====
describe:When keepdim >= total singular values (smidx == 0), nothing is dropped. return_err
must still return a zero error tensor rather than crashing or returning garbage.
====================*/
TEST_F(linalg_Test, BkUt_Svd_truncate_return_err_no_truncation) {
  // keepdim=999 keeps everything; return_err=1 is a scalar zero, while return_err=2 keeps the
  // legacy one-element zero tensor.
  for (unsigned int re : {1u, 2u}) {
    std::vector<UniTensor> res = linalg::Svd_truncate(svd_T, 999, 0, true, re);
    ASSERT_EQ(res.size(), 4u) << "return_err=" << re;
    if (re == 1) {
      ASSERT_TRUE(res[3].shape().empty()) << "return_err=" << re;
      EXPECT_EQ(res[3].at({}), Scalar(0.0)) << "return_err=" << re;
    } else {
      ASSERT_EQ(res[3].shape()[0], 1u) << "return_err=" << re;
      EXPECT_EQ(res[3].at({0}), Scalar(0.0)) << "return_err=" << re;
    }
  }
}

TEST_F(linalg_Test, BkUt_Gesvd_truncate_return_err_no_truncation) {
  for (unsigned int re : {1u, 2u}) {
    std::vector<UniTensor> res = linalg::Gesvd_truncate(svd_T, 999, 0, true, true, re);
    ASSERT_EQ(res.size(), 4u) << "return_err=" << re;
    if (re == 1) {
      ASSERT_TRUE(res[3].shape().empty()) << "return_err=" << re;
      EXPECT_EQ(res[3].at({}), Scalar(0.0)) << "return_err=" << re;
    } else {
      ASSERT_EQ(res[3].shape()[0], 1u) << "return_err=" << re;
      EXPECT_EQ(res[3].at({0}), Scalar(0.0)) << "return_err=" << re;
    }
  }
}

/*=====test info=====
describe:BlockFermionic truncated SVD: return_err must return the discarded singular values
in descending order (return_err>1) or just the largest discarded value (return_err==1),
matching the same contract as the Block (non-fermionic) path.
====================*/
TEST_F(linalg_Test, BkFermionicUt_Svd_truncate_return_err_returns_discarded_values) {
  UniTensor fermi_T = make_square_fermionic({"a", "b"});
  random::uniform_(fermi_T, -1.0, 1.0, 42);
  ASSERT_EQ(fermi_T.uten_type(), UTenType.BlockFermionic);

  std::vector<UniTensor> full = linalg::Svd_truncate(fermi_T, 999, 0, true, 0);
  std::vector<UniTensor> trunc = linalg::Svd_truncate(fermi_T, 1, 0, true, 999);
  Tensor all_svals = SortedBlockSingularValues(full[0]);

  ASSERT_GE(trunc.size(), 4u);
  ASSERT_GE(all_svals.shape()[0], trunc[0].shape()[0] + trunc[3].shape()[0]);
  for (cytnx_uint64 i = 0; i < trunc[3].shape()[0]; i++) {
    EXPECT_EQ(all_svals.at({trunc[3].shape()[0] - 1 - i}), trunc[3].at({i}));
  }

  // return_err=1: scalar equal to largest discarded
  std::vector<UniTensor> trunc1 = linalg::Svd_truncate(fermi_T, 1, 0, true, 1);
  ASSERT_TRUE(trunc1.back().shape().empty());
  EXPECT_EQ(trunc1.back().at({}), trunc[3].at({0}));
}

TEST_F(linalg_Test, BkFermionicUt_Gesvd_truncate_return_err_returns_discarded_values) {
  UniTensor fermi_T = make_square_fermionic({"a", "b"});
  random::uniform_(fermi_T, -1.0, 1.0, 42);
  ASSERT_EQ(fermi_T.uten_type(), UTenType.BlockFermionic);

  std::vector<UniTensor> full = linalg::Gesvd_truncate(fermi_T, 999, 0, true, true, 0);
  std::vector<UniTensor> trunc = linalg::Gesvd_truncate(fermi_T, 1, 0, true, true, 999);
  Tensor all_svals = SortedBlockSingularValues(full[0]);

  ASSERT_GE(trunc.size(), 4u);
  ASSERT_GE(all_svals.shape()[0], trunc[0].shape()[0] + trunc[3].shape()[0]);
  for (cytnx_uint64 i = 0; i < trunc[3].shape()[0]; i++) {
    EXPECT_EQ(all_svals.at({trunc[3].shape()[0] - 1 - i}), trunc[3].at({i}));
  }

  // return_err=1: scalar equal to largest discarded
  std::vector<UniTensor> trunc1 = linalg::Gesvd_truncate(fermi_T, 1, 0, true, true, 1);
  ASSERT_TRUE(trunc1.back().shape().empty());
  EXPECT_EQ(trunc1.back().at({}), trunc[3].at({0}));
}

// TEST_F(linalg_Test, BkUt_Svd_truncate3) {
//   Bond I = Bond(BD_IN, {Qs(-5), Qs(-3), Qs(-1), Qs(1), Qs(3), Qs(5)}, {1, 4, 10, 9, 5, 1});
//   Bond J = Bond(BD_OUT, {Qs(1), Qs(-1)}, {1, 1});
//   Bond K = Bond(BD_OUT, {Qs(1), Qs(-1)}, {1, 1});
//   Bond L = Bond(BD_OUT, {Qs(-5), Qs(-3), Qs(-1), Qs(1), Qs(3), Qs(5)}, {1, 4, 10, 9, 5, 1});
//   UniTensor cyT = UniTensor({I, J, K, L}, {"a", "b", "c", "d"}, 2, Type.Double, Device.cpu,
//   false); auto cyT2 = UniTensor::Load(data_dir + "Svd_truncate/Svd_truncate2.cytnx");
//   std::vector<UniTensor> res = linalg::Svd_truncate(cyT, 30, 0, true);
//   auto con_T1 = Contract(Contract(res[2], res[0]), res[1]);
//   auto con_T2 = Contract(Contract(res[1], res[0]), res[2]);
// }

// TEST_F(linalg_Test, BkUt_Svd_truncate4) {
//   Bond I = Bond(BD_IN, {Qs(-4), Qs(-2), Qs(0), Qs(2), Qs(4)}, {2, 7, 10, 8, 3});
//   Bond J = Bond(BD_OUT, {Qs(1), Qs(-1)}, {1, 1});
//   Bond K = Bond(BD_OUT, {Qs(1), Qs(-1)}, {1, 1});
//   Bond L = Bond(BD_OUT, {Qs(-4), Qs(-2), Qs(0), Qs(2), Qs(4), Qs(6)}, {1, 5, 10, 9, 4, 1});
//   UniTensor cyT = UniTensor({I, J, K, L}, {"a", "b", "c", "d"}, 2, Type.Double, Device.cpu,
//   false); cyT = UniTensor::Load(data_dir + "Svd_truncate/Svd_truncate3.cytnx");
//   std::vector<UniTensor> res = linalg::Svd_truncate(cyT, 30, 0, true);
//   auto con_T1 = Contract(Contract(res[2], res[0]), res[1]);
//   auto con_T2 = Contract(Contract(res[1], res[0]), res[2]);
// }

TEST_F(linalg_Test, BkUt_Qr1) {
  auto res = linalg::Qr(H);
  auto Q = res[0];
  auto R = res[1];
  for (size_t i = 0; i < 27; i++)
    for (size_t j = 0; j < 27; j++) {
      if (R.elem_exists({i, j})) {
        EXPECT_TRUE(abs((double)(R.at({i, j}).real()) - (double)(Qr_Rans.at({i, j}).real())) <
                    1E-12);
        // EXPECT_EQ((double)(R.at({i,j}).real()),(double)(Qr_Rans.at({i,j}).real()));
      }
      if (Q.elem_exists({i, j})) {
        EXPECT_TRUE(abs((double)(Q.at({i, j}).real()) - (double)(Qr_Qans.at({i, j}).real())) <
                    1E-12);
        // EXPECT_EQ((double)(Q.at({i,j}).real()),(double)(Qr_Qans.at({i,j}).real()));
      }
    }
}

// Check if QR works when the first bond is BD_OUT.
// Q R must recompose to the (multi-sector) input.
TEST_F(linalg_Test, BkUt_Qr_reversed_qnums) {
  UniTensor T({Bond(BD_OUT, {Qs(0) >> 1, Qs(1) >> 1}, {Symmetry::U1()}),
               Bond(BD_OUT, {Qs(0) >> 1, Qs(1) >> 1}, {Symmetry::U1()}),
               Bond(BD_IN, {Qs(0) >> 2, Qs(1) >> 2, Qs(2) >> 2}, {Symmetry::U1()})},
              {"a", "b", "c"}, 2, Type.Double, Device.cpu, false);
  random::uniform_(T, -1.0, 1.0, 0);
  auto res = linalg::Qr(T);
  EXPECT_GT(res[0].bonds().back().qnums().size(), 1u);  // multi-sector auxiliary bond
  EXPECT_TRUE((T - Contract(res[0], res[1])).Norm().item() < 1e-9);
}

// 4-leg square Hermitian Block UniTensor where the two row legs use the directions given by input
// arguments. Real-valued (row,col)-symmetric values -> Hermitian.
inline UniTensor make_rank4_hermitian_left_dirs(bondType first, bondType second) {
  Bond a = Bond(first, {Qs(0) >> 1, Qs(1) >> 1}, {Symmetry::U1()});
  Bond b = Bond(second, {Qs(0) >> 1, Qs(1) >> 1}, {Symmetry::U1()});
  UniTensor M = UniTensor({a, b, a.redirect(), b.redirect()}, {"a", "b", "c", "d"});
  M.set_rowrank_(2);
  auto sh = M.shape();
  for (cytnx_uint64 i = 0; i < sh[0]; i++)
    for (cytnx_uint64 j = 0; j < sh[1]; j++)
      for (cytnx_uint64 k = 0; k < sh[2]; k++)
        for (cytnx_uint64 l = 0; l < sh[3]; l++) {
          auto p = M.at({i, j, k, l});
          if (p.exists()) {
            cytnx_uint64 row = i * sh[1] + j;
            cytnx_uint64 col = k * sh[3] + l;
            p = (row == col) ? double(1 + row) : double(0.5 * (row + col + 1));
          }
        }
  return M;
}

// Eigh and ExpM use BD_OUT/BD_OUT as left bonds; Eig and ExpH use BD_OUT/BD_IN; Covers different
// combine_bonds patternes in the linalg functions.

TEST_F(linalg_Test, BkUt_Eigh_reversed_qnums) {
  const double tol = 1e-10;
  UniTensor M = make_rank4_hermitian_left_dirs(BD_OUT, BD_OUT);
  auto out = linalg::Eigh(M);
  ASSERT_EQ(out.size(), 2u);
  UniTensor e = out[0], V = out[1];
  EXPECT_TRUE(e.is_diag());
  EXPECT_EQ(e.dtype(), Type.Double);
  EXPECT_GT(V.bonds().back().qnums().size(), 1u);  // multi-sector auxiliary bond
  // eigenvectors are orthonormal: V^dagger V = I
  expect_unitary(V, "_aux_L", tol);
  // M = V * diag(e) * V^dagger. V's row legs are M's row labels ("a","b") and its col is "_aux_L";
  // rename V.Dagger()'s row labels to M's col labels and its aux to "_aux_R" to chain correctly.
  UniTensor Vdag = V.Dagger();
  Vdag.relabel_("_aux_L", "_aux_R");
  Vdag.relabel_("a", "c");
  Vdag.relabel_("b", "d");
  UniTensor reconstructed = Contract(Contract(V, e), Vdag);
  EXPECT_TRUE((M - reconstructed.permute_(M.labels())).Norm().item() < tol);
}

TEST_F(linalg_Test, BkUt_Eig_reversed_qnums) {
  const double tol = 1e-10;
  UniTensor M = make_rank4_hermitian_left_dirs(BD_OUT, BD_IN);
  auto out = linalg::Eig(M);
  ASSERT_EQ(out.size(), 2u);
  EXPECT_TRUE(out[0].is_diag());
  EXPECT_GT(out[1].bonds().back().qnums().size(), 1u);  // multi-sector auxiliary bond
  // M is Hermitian -> Eig and Eigh must return the same (real) spectrum.
  expect_same_diagonal(out[0], linalg::Eigh(M)[0], tol);
}

TEST_F(linalg_Test, BkUt_ExpH_reversed_qnums) {
  const double tol = 1e-10;
  const double alpha = 0.1;
  UniTensor M = make_rank4_hermitian_left_dirs(BD_OUT, BD_IN);
  UniTensor R = linalg::ExpH(M, alpha);
  // exp(alpha*M) commutes with M
  UniTensor RM = Contract(R.relabel({"a", "b", "_m", "_n"}), M.relabel({"_m", "_n", "c", "d"}));
  UniTensor MR = Contract(M.relabel({"a", "b", "_m", "_n"}), R.relabel({"_m", "_n", "c", "d"}));
  EXPECT_TRUE((RM - MR.permute_(RM.labels())).Norm().item() < tol);
  // spectrum of ExpH(M, alpha) is exp(alpha * spectrum(M))
  expect_exp_spectrum(linalg::Eigh(R)[0], linalg::Eigh(M)[0], alpha, tol);
}

TEST_F(linalg_Test, BkUt_ExpM_reversed_qnums) {
  const double tol = 1e-10;
  const double alpha = 0.1;
  UniTensor M = make_rank4_hermitian_left_dirs(BD_OUT, BD_OUT);
  UniTensor R = linalg::ExpM(M, alpha);
  // exp(alpha*M) commutes with M
  UniTensor RM = Contract(R.relabel({"a", "b", "_m", "_n"}), M.relabel({"_m", "_n", "c", "d"}));
  UniTensor MR = Contract(M.relabel({"a", "b", "_m", "_n"}), R.relabel({"_m", "_n", "c", "d"}));
  EXPECT_TRUE((RM - MR.permute_(RM.labels())).Norm().item() < tol);
  // M is Hermitian, so ExpM(M, alpha) has the same spectrum as ExpH(M, alpha): exp(alpha *
  // spec(M)).
  expect_exp_spectrum(linalg::Eigh(R)[0], linalg::Eigh(M)[0], alpha, tol);
}

TEST_F(linalg_Test, BkUt_expH) {
  auto res = linalg::ExpH(H);
  for (size_t i = 0; i < 27; i++)
    for (size_t j = 0; j < 27; j++) {
      if (res.elem_exists({i, j})) {
        EXPECT_TRUE(abs((double)(res.at({i, j}).real()) - (double)(expH_ans.at({i, j}).real())) <
                    1E-8);
        // EXPECT_EQ((double)(R.at({i,j}).real()),(double)(Qr_Rans.at({i,j}).real()));
      }
    }
}

TEST_F(linalg_Test, BkUt_expM) {
  auto res = linalg::ExpM(H);
  for (size_t i = 0; i < 27; i++)
    for (size_t j = 0; j < 27; j++) {
      if (res.elem_exists({i, j})) {
        EXPECT_TRUE(abs((double)(res.at({i, j}).real()) - (double)(expH_ans.at({i, j}).real())) <
                    1E-8);
        // EXPECT_EQ((double)(R.at({i,j}).real()),(double)(Qr_Rans.at({i,j}).real()));
      }
    }
}

TEST_F(linalg_Test, DenseUt_Gesvd_truncate) {
  std::vector<UniTensor> full = linalg::Gesvd_truncate(svd_T_dense, 999, 0, true, true, 999);
  EXPECT_EQ(full[0].shape()[0], 11);

  EXPECT_EQ(full[1].shape()[0], 11);
  EXPECT_EQ(full[1].shape()[1], 11);

  EXPECT_EQ(full[2].shape()[0], 11);
  EXPECT_EQ(full[2].shape()[1], 13);

  std::vector<UniTensor> truc1 = linalg::Gesvd_truncate(svd_T_dense, 5, 0, true, true, 999);

  EXPECT_EQ(truc1[0].shape()[0], 5);

  EXPECT_EQ(truc1[1].shape()[0], 11);
  EXPECT_EQ(truc1[1].shape()[1], 5);

  EXPECT_EQ(truc1[2].shape()[0], 5);
  EXPECT_EQ(truc1[2].shape()[1], 13);

  EXPECT_EQ(truc1[3].shape()[0], 6);

  for (size_t i = 0; i < 5; i++) {
    EXPECT_EQ(full[0].at({i}), truc1[0].at({i}));
  }
  for (size_t i = 0; i < 6; i++) {
    EXPECT_EQ(full[0].at({i + 5}), truc1[3].at({i}));
  }

  for (size_t i = 0; i < 11; i++) {
    for (size_t j = 0; j < 5; j++) {
      EXPECT_EQ(full[1].at({i, j}), truc1[1].at({i, j}));
    }
  }
  for (size_t i = 0; i < 5; i++) {
    for (size_t j = 0; j < 13; j++) {
      EXPECT_EQ(full[2].at({i, j}), truc1[2].at({i, j}));
    }
  }

  std::vector<UniTensor> truc2 = linalg::Gesvd_truncate(svd_T_dense, 5, 1e-12, true, true, 999);

  EXPECT_EQ(truc2[0].shape()[0], 2);

  EXPECT_EQ(truc2[1].shape()[0], 11);
  EXPECT_EQ(truc2[1].shape()[1], 2);

  EXPECT_EQ(truc2[2].shape()[0], 2);
  EXPECT_EQ(truc2[2].shape()[1], 13);

  EXPECT_EQ(truc2[3].shape()[0], 9);

  for (size_t i = 0; i < 2; i++) {
    EXPECT_EQ(full[0].at({i}), truc2[0].at({i}));
  }
  for (size_t i = 0; i < 9; i++) {
    EXPECT_EQ(full[0].at({i + 2}), truc2[3].at({i}));
  }

  for (size_t i = 0; i < 11; i++) {
    for (size_t j = 0; j < 2; j++) {
      EXPECT_EQ(full[1].at({i, j}), truc2[1].at({i, j}));
    }
  }
  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 13; j++) {
      EXPECT_EQ(full[2].at({i, j}), truc2[2].at({i, j}));
    }
  }
}

TEST_F(linalg_Test, DenseUt_Svd_truncate) {
  std::vector<UniTensor> full = linalg::Svd_truncate(svd_T_dense, 999, 0, true, 999);
  EXPECT_EQ(full[0].shape()[0], 11);

  EXPECT_EQ(full[1].shape()[0], 11);
  EXPECT_EQ(full[1].shape()[1], 11);

  EXPECT_EQ(full[2].shape()[0], 11);
  EXPECT_EQ(full[2].shape()[1], 13);

  std::vector<UniTensor> truc1 = linalg::Svd_truncate(svd_T_dense, 5, 0, true, 999);

  EXPECT_EQ(truc1[0].shape()[0], 5);

  EXPECT_EQ(truc1[1].shape()[0], 11);
  EXPECT_EQ(truc1[1].shape()[1], 5);

  EXPECT_EQ(truc1[2].shape()[0], 5);
  EXPECT_EQ(truc1[2].shape()[1], 13);

  EXPECT_EQ(truc1[3].shape()[0], 6);

  for (size_t i = 0; i < 5; i++) {
    EXPECT_EQ(full[0].at({i}), truc1[0].at({i}));
  }
  for (size_t i = 0; i < 6; i++) {
    EXPECT_EQ(full[0].at({i + 5}), truc1[3].at({i}));
  }

  for (size_t i = 0; i < 11; i++) {
    for (size_t j = 0; j < 5; j++) {
      EXPECT_EQ(full[1].at({i, j}), truc1[1].at({i, j}));
    }
  }
  for (size_t i = 0; i < 5; i++) {
    for (size_t j = 0; j < 13; j++) {
      EXPECT_EQ(full[2].at({i, j}), truc1[2].at({i, j}));
    }
  }

  std::vector<UniTensor> truc2 = linalg::Svd_truncate(svd_T_dense, 5, 1e-12, true, 999);

  EXPECT_EQ(truc2[0].shape()[0], 2);

  EXPECT_EQ(truc2[1].shape()[0], 11);
  EXPECT_EQ(truc2[1].shape()[1], 2);

  EXPECT_EQ(truc2[2].shape()[0], 2);
  EXPECT_EQ(truc2[2].shape()[1], 13);

  EXPECT_EQ(truc2[3].shape()[0], 9);

  for (size_t i = 0; i < 2; i++) {
    EXPECT_EQ(full[0].at({i}), truc2[0].at({i}));
  }
  for (size_t i = 0; i < 9; i++) {
    EXPECT_EQ(full[0].at({i + 2}), truc2[3].at({i}));
  }

  for (size_t i = 0; i < 11; i++) {
    for (size_t j = 0; j < 2; j++) {
      EXPECT_EQ(full[1].at({i, j}), truc2[1].at({i, j}));
    }
  }
  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 13; j++) {
      EXPECT_EQ(full[2].at({i, j}), truc2[2].at({i, j}));
    }
  }
}

TEST_F(linalg_Test, DenseUt_Pow) {
  UniTensor Ht = UniTensor(A);
  auto res = linalg::Pow(Ht, 3);
  for (size_t i = 0; i < 9; i++)
    for (size_t j = 0; j < 9; j++) {
      // if(res.elem_exists({i,j})){
      EXPECT_TRUE(abs((double)(res.at({i, j}).real()) - (double)(Pow_ans.at({i, j}).real())) <
                  1E-8);
      // EXPECT_EQ((double)(R.at({i,j}).real()),(double)(Qr_Rans.at({i,j}).real()));
      //}
    }
}

TEST_F(linalg_Test, DenseUt_Pow_) {
  UniTensor Ht = UniTensor(A);
  linalg::Pow_(Ht, 3);
  for (size_t i = 0; i < 9; i++)
    for (size_t j = 0; j < 9; j++) {
      // if(Ht.elem_exists({i,j})){
      EXPECT_TRUE(abs((double)(Ht.at({i, j}).real()) - (double)(Pow_ans.at({i, j}).real())) < 1E-8);
      // EXPECT_EQ((double)(R.at({i,j}).real()),(double)(Qr_Rans.at({i,j}).real()));
      //}
    }
}

TEST_F(linalg_Test, DenseUt_Mod) {
  UniTensor At = UniTensor(A);
  auto res = linalg::Mod(100 * At, 3);
  for (size_t i = 0; i < 9; i++)
    for (size_t j = 0; j < 9; j++) {
      // if(Ht.elem_exists({i,j})){
      EXPECT_TRUE(abs((double)(res.at({i, j}).real()) - (double)(Mod_ans.at({i, j}).real())) <
                  1E-8);
      // EXPECT_EQ((double)(R.at({i,j}).real()),(double)(Qr_Rans.at({i,j}).real()));
      //}
    }
}

TEST_F(linalg_Test, Tensor_Gemm) {
  Tensor res_d = linalg::Gemm(0.5, arange3x3d, eye3x3d);
  Tensor ans_d = arange3x3d * 0.5;

  for (size_t i = 0; i < 3; i++)
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ(res_d(i, j).item(), ans_d(i, j).item());
    }

  Tensor res_cd = linalg::Gemm(0.5, arange3x3cd, eye3x3cd);
  Tensor ans_cd = arange3x3cd * 0.5;

  for (size_t i = 0; i < 3; i++)
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ(res_cd(i, j).item().real(), ans_cd(i, j).item().real());
      EXPECT_EQ(res_cd(i, j).item().imag(), ans_cd(i, j).item().imag());
    }
}

TEST_F(linalg_Test, Tensor_Gemm_) {
  Tensor C_d = arange3x3d.clone();
  linalg::Gemm_(1, arange3x3d, eye3x3d, 0.5, C_d);
  Tensor ans_d = arange3x3d * 1.5;
  for (size_t i = 0; i < 3; i++)
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ(C_d(i, j).item(), ans_d(i, j).item());
    }

  Tensor C_cd = arange3x3cd.clone();
  linalg::Gemm_(1, arange3x3cd, eye3x3cd, 0.5, C_cd);
  Tensor ans_cd = arange3x3cd * 1.5;

  for (size_t i = 0; i < 3; i++)
    for (size_t j = 0; j < 3; j++) {
      EXPECT_EQ(C_cd(i, j).item().real(), ans_cd(i, j).item().real());
      EXPECT_EQ(C_cd(i, j).item().imag(), ans_cd(i, j).item().imag());
    }
}

TEST_F(linalg_Test, Tensor_Norm) {
  cytnx_double ans = 0;
  for (cytnx_uint64 i = 0; i < 9; i++) {
    ans += i * i * 2;
  }
  ans = sqrt(ans);
  EXPECT_EQ(linalg::Norm(arange3x3cd).item(), ans);
}

TEST_F(linalg_Test, Tensor_Add_mixed_dtype_type_promote_cpu) {
  Tensor lhs = arange(0, 4, 1, Type.Uint32).reshape(2, 2);
  Tensor rhs = arange(0, 4, 1, Type.Int16).reshape(2, 2);

  Tensor out = linalg::Add(lhs, rhs);
  EXPECT_EQ(out.dtype(), Type.type_promote(lhs.dtype(), rhs.dtype()));
  EXPECT_EQ((cytnx_int64)out(1, 1).item().real(), 6);
}

TEST_F(linalg_Test, Tensor_Add_scalar_mixed_dtype_type_promote_cpu) {
  Tensor rhs = arange(0, 4, 1, Type.Int16).reshape(2, 2);
  const cytnx_uint32 lhs_scalar = 5;

  Tensor out_lhs = linalg::Add(lhs_scalar, rhs);
  Tensor out_rhs = linalg::Add(rhs, lhs_scalar);
  const unsigned int promoted = Type.type_promote(Type.Uint32, rhs.dtype());

  EXPECT_EQ(out_lhs.dtype(), promoted);
  EXPECT_EQ(out_rhs.dtype(), promoted);
  EXPECT_EQ((cytnx_int64)out_lhs(1, 1).item().real(), 8);
  EXPECT_EQ((cytnx_int64)out_rhs(1, 1).item().real(), 8);
}

TEST_F(linalg_Test, DenseUt_Norm) {
  cytnx_double ans = 0;
  for (cytnx_uint64 i = 0; i < 9; i++) {
    ans += i * i * 2;
  }
  ans = sqrt(ans);
  EXPECT_EQ(linalg::Norm(arange3x3cd_ut).item(), ans);
}

TEST_F(linalg_Test, BkUt_Norm) {
  cytnx_double ans = 0;
  for (cytnx_uint64 i = 0; i < 9; i++) {
    ans += i * i * 2;
  }
  ans += 9;
  ans = sqrt(ans);
  Bond I = Bond(BD_IN, {Qs(-1), Qs(1)}, {3, 3});
  Bond J = Bond(BD_OUT, {Qs(-1), Qs(1)}, {3, 3});
  UniTensor in = UniTensor({I, J});
  auto cd_in = in.astype(Type.ComplexDouble);
  cd_in.put_block_(arange3x3cd, 0);
  cd_in.put_block_(ones3x3cd, 1);
  // EXPECT_EQ(cytnx_double(linalg::Norm(in).item().real()), ans);
  EXPECT_TRUE(abs(cytnx_double(linalg::Norm(cd_in).item().real()) - ans) <
              1e-13);  // not sure why some precision lost.
}

TEST_F(linalg_Test, Tensor_Eig) {
  auto res = linalg::Eig(arange3x3cd);
  auto e = UniTensor(res[0], true);
  e.relabel_({"a", "b"});
  auto v = UniTensor(res[1]);
  v.relabel_({"i", "a"});
  auto vt = UniTensor(linalg::InvM(v.get_block()));
  vt.relabel_({"b", "j"});
  EXPECT_TRUE((UniTensor(arange3x3cd) - Contract(Contract(e, v), vt)).Norm().item() < 1e-13);
}

TEST_F(linalg_Test, Tensor_Eig_RowV) {
  const double tol = 1e-13;
  auto row_v = linalg::Eig(arange3x3cd, true, true);
  // auto col_v = linalg::Eig(arange3x3cd, true, false);

  // EXPECT_TRUE(AreNearlyEqTensor(row_v[0], col_v[0], tol));
  // EXPECT_TRUE(AreNearlyEqTensor(row_v[1], linalg::InvM(col_v[1]), tol));

  auto e = UniTensor(row_v[0], true);
  e.relabel_({"a", "b"});
  auto v = UniTensor(row_v[1].Conj().permute({1, 0}));
  v.relabel_({"i", "a"});
  auto vt = UniTensor(linalg::InvM(v.get_block()));
  vt.relabel_({"b", "j"});
  EXPECT_TRUE((UniTensor(arange3x3cd) - Contract(Contract(e, v), vt)).Norm().item() < tol);
}

TEST_F(linalg_Test, Tensor_Eig_ValuesOnly) {
  auto values_only = linalg::Eig(arange3x3cd, false);
  auto with_vectors = linalg::Eig(arange3x3cd, true);

  ASSERT_EQ(values_only.size(), 1);
  ASSERT_EQ(with_vectors.size(), 2);
  EXPECT_EQ(values_only[0].dtype(), Type.ComplexDouble);
  EXPECT_EQ(values_only[0].shape(), with_vectors[0].shape());
  std::vector<bool> matched(with_vectors[0].shape()[0], false);
  for (cytnx_uint64 i = 0; i < values_only[0].shape()[0]; ++i) {
    bool found = false;
    const auto value = values_only[0].at<cytnx_complex128>({i});
    for (cytnx_uint64 j = 0; j < with_vectors[0].shape()[0]; ++j) {
      if (!matched[j] && std::abs(value - with_vectors[0].at<cytnx_complex128>({j})) < 1e-13) {
        matched[j] = true;
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found);
  }
}

TEST_F(linalg_Test, Tensor_Eigh) {
  auto her = arange3x3cd + arange3x3cd.Conj().permute({1, 0});
  auto res = linalg::Eigh(her);
  auto e = UniTensor(res[0], true);
  e.relabel_({"a", "b"});
  auto v = UniTensor(res[1]);
  v.relabel_({"i", "a"});
  auto vt = UniTensor(linalg::InvM(v.get_block()));
  vt.relabel_({"b", "j"});
  EXPECT_TRUE((UniTensor(her) - Contract(Contract(e, v), vt)).Norm().item() < 1e-13);
}

TEST_F(linalg_Test, Tensor_Eigh_RowV) {
  const double tol = 1e-13;
  auto her = arange3x3cd + arange3x3cd.Conj().permute({1, 0});
  auto row_v = linalg::Eigh(her, true, true);
  auto col_v = linalg::Eigh(her, true, false);

  EXPECT_TRUE(AreNearlyEqTensor(row_v[0], col_v[0], tol));
  EXPECT_TRUE(AreNearlyEqTensor(row_v[1], linalg::InvM(col_v[1]), tol));

  auto e = UniTensor(row_v[0], true);
  e.relabel_({"a", "b"});
  auto v = UniTensor(row_v[1].Conj().permute({1, 0}));
  v.relabel_({"i", "a"});
  auto vt = UniTensor(linalg::InvM(v.get_block()));
  vt.relabel_({"b", "j"});
  EXPECT_TRUE((UniTensor(her) - Contract(Contract(e, v), vt)).Norm().item() < tol);
}

TEST_F(linalg_Test, DenseUt_Eig) {
  auto res = linalg::Eig(arange3x3cd_ut);
  auto e = res[0];
  e.relabel_({"a", "b"});
  auto v = res[1];
  v.relabel_({"i", "a"});
  auto vt = UniTensor(linalg::InvM(v.get_block()));
  vt.relabel_({"b", "j"});
  EXPECT_TRUE((UniTensor(arange3x3cd) - Contract(Contract(e, v), vt)).Norm().item() < 1e-13);
}

TEST_F(linalg_Test, DenseUt_Eig_RowV) {
  const double tol = 1e-13;
  auto row_v = linalg::Eig(arange3x3cd_ut, true, true);
  // auto col_v = linalg::Eig(arange3x3cd_ut, true, false);

  // EXPECT_TRUE(AreNearlyEqUniTensor(row_v[0], col_v[0], tol));
  // EXPECT_TRUE(AreNearlyEqUniTensor(row_v[1], linalg::InvM(col_v[1]), tol));

  auto e = row_v[0];
  e.relabel_({"a", "b"});
  auto v = UniTensor(row_v[1].get_block_().Conj().permute({1, 0}));
  v.relabel_({"i", "a"});
  auto vt = UniTensor(linalg::InvM(v.get_block()));
  vt.relabel_({"b", "j"});
  EXPECT_TRUE((UniTensor(arange3x3cd) - Contract(Contract(e, v), vt)).Norm().item() < tol);
}

TEST_F(linalg_Test, DenseUt_Eigh) {
  const double tol = 1e-13;
  auto her = arange3x3cd + arange3x3cd.Conj().permute({1, 0});
  auto res = linalg::Eigh(UniTensor(her));
  auto e = res[0];
  e.relabel_({"a", "b"});
  auto v = res[1];
  v.relabel_({"i", "a"});
  auto vt = UniTensor(linalg::InvM(v.get_block()));
  vt.relabel_({"b", "j"});
  EXPECT_TRUE((UniTensor(her) - Contract(Contract(e, v), vt)).Norm().item() < tol);
}

TEST_F(linalg_Test, DenseUt_Eigh_RowV) {
  const double tol = 1e-13;
  auto her = arange3x3cd + arange3x3cd.Conj().permute({1, 0});
  auto row_v = linalg::Eigh(UniTensor(her), true, true);
  auto col_v = linalg::Eigh(UniTensor(her), true, false);

  EXPECT_TRUE(AreNearlyEqUniTensor(row_v[0], col_v[0], tol));
  EXPECT_TRUE(AreNearlyEqUniTensor(row_v[1], linalg::InvM(col_v[1]), tol));

  auto e = row_v[0];
  e.relabel_({"a", "b"});
  auto v = UniTensor(row_v[1].get_block_().Conj().permute({1, 0}));
  v.relabel_({"i", "a"});
  auto vt = UniTensor(linalg::InvM(v.get_block()));
  vt.relabel_({"b", "j"});
  EXPECT_TRUE((UniTensor(her) - Contract(Contract(e, v), vt)).Norm().item() < tol);
}

TEST_F(linalg_Test, DenseUt_Eig_RowrankMustBeLessThanRank) {
  UniTensor invalid = UniTensor(arange(0, 4, 1, Type.ComplexDouble).reshape(2, 2), false, 2);
  EXPECT_THROW({ linalg::Eig(invalid); }, std::logic_error);
}

TEST_F(linalg_Test, DenseUt_Eigh_RowrankMustBeLessThanRank) {
  auto her = arange3x3cd + arange3x3cd.Conj().permute({1, 0});
  UniTensor invalid = UniTensor(her, false, 2);
  EXPECT_THROW({ linalg::Eigh(invalid); }, std::logic_error);
}

// TEST_F(linalg_Test, Tensor_Inv) { EXPECT_TRUE(false); }

// TEST_F(linalg_Test, Tensor_Inv_) { EXPECT_TRUE(false); }

// TEST_F(linalg_Test, DenseUt_Inv) { EXPECT_TRUE(false); }

// TEST_F(linalg_Test, DenseUt_Inv_) { EXPECT_TRUE(false); }

TEST_F(linalg_Test, Tensor_InvM) {
  auto inv = linalg::InvM(invertable3x3cd);
  EXPECT_TRUE((linalg::Tensordot(invertable3x3cd, inv, {1}, {0}) - eye3x3cd).Norm().item() < 1e-13);
}

TEST_F(linalg_Test, Tensor_InvM_) {
  auto inv = invertable3x3cd.clone();
  linalg::InvM_(inv);
  EXPECT_TRUE((linalg::Tensordot(invertable3x3cd, inv, {1}, {0}) - eye3x3cd).Norm().item() < 1e-13);
}

TEST_F(linalg_Test, DenseUt_InvM) {
  auto inv = linalg::InvM(invertable3x3cd_ut);
  inv.relabel_({"1", "2"});  // invertable3x3cd_ut is labeled "0","1".
  EXPECT_TRUE((invertable3x3cd_ut.contract(inv) - UniTensor(eye3x3cd)).Norm().item() < 1e-13);
}

TEST_F(linalg_Test, DenseUt_InvM_) {
  auto inv = invertable3x3cd_ut.clone();
  inv.relabel_({"1", "2"});  // invertable3x3cd_ut is labeled "0","1".
  linalg::InvM_(inv);
  EXPECT_TRUE((invertable3x3cd_ut.contract(inv) - UniTensor(eye3x3cd)).Norm().item() < 1e-13);
}

// TEST_F(linalg_Test, DenseUt_Mod_UtUt){
//     UniTensor At = UniTensor(A);
//     UniTensor Bt = UniTensor(B);
//     auto res = linalg::Mod(100*At, Bt);
//     for(size_t i = 0;i<9;i++)
//       for(size_t j = 0; j<9;j++){
//           //if(Ht.elem_exists({i,j})){
//             EXPECT_TRUE(abs((double)(res.at({i,j}).real())-(double)(ModUtUt_ans.at({i,j}).real()))
//             < 1E-8);
//             //EXPECT_EQ((double)(R.at({i,j}).real()),(double)(Qr_Rans.at({i,j}).real()));
//           //}
//       }
// }

/*=====test info=====
describe:test SVD unitarity and reconstruction for fermionic tensors with mixed in/out legs
====================*/
TEST_F(linalg_Test, BkFUt_SvdUnitaryAndReconstruction) {
  const double tol = 1e-10;
  UniTensor M = make_mixed_inout_fermionic();
  // U, vT are isometries and M = U S vT (direct SVD reconstruction).
  expect_svd_reconstructs(linalg::Svd_truncate(M, 1000, 0., true, 0), M, tol);
}

/*=====test info=====
describe:Eigh for fermionic tensors: real spectrum, orthonormal eigenvectors
  (V^dagger V = I), and spectrum invariance under a consistent sign-flip permute.
====================*/
TEST_F(linalg_Test, BkFUt_Eigh) {
  const double tol = 1e-10;
  UniTensor H = make_square_fermionic({"i", "o"});
  // real symmetric per sector -> Hermitian fermionic UniTensor
  H.at({0, 0}) = 2.0;
  H.at({0, 1}) = 0.7;
  H.at({1, 0}) = 0.7;
  H.at({1, 1}) = 3.0;
  H.at({2, 2}) = 1.5;
  H.at({2, 3}) = -0.4;
  H.at({3, 2}) = -0.4;
  H.at({3, 3}) = 2.5;

  auto out = linalg::Eigh(H);
  ASSERT_EQ(out.size(), 2);
  UniTensor e = out[0], V = out[1];
  EXPECT_TRUE(e.is_diag());
  EXPECT_EQ(e.dtype(), Type.Double);  // Eigh eigenvalues are real

  // eigenvectors are orthonormal: V^dagger V = I
  expect_unitary(V, "_aux_L", tol);

  // cross-check against the general solver: for a Hermitian operator, Eig must return the same
  // (real) spectrum as Eigh (Eig eigenvalues are complex; sorted_diagonal compares real parts).
  expect_same_diagonal(e, linalg::Eig(H)[0], tol);

  // sign-flip coverage: a consistent permute makes the pending signflip non-trivial (exercising
  // the negation path in Eigh_BlockFermionic_UT_internal). Verify the eigenvectors are still
  // orthonormal (V^dagger V = I) and the spectrum is unchanged. Run real and genuinely
  // complex-Hermitian so both the real (dsyev) and complex (zheev) Eigh code paths are covered.
  for (bool cplx : {false, true}) {
    SCOPED_TRACE(cplx ? "ComplexDouble" : "Double");
    UniTensor M = make_rank4_herm_dt(cplx);
    UniTensor Mp = permute_with_signflips(M);
    auto outp = linalg::Eigh(Mp);
    UniTensor ep = outp[0], Vp = outp[1];
    EXPECT_EQ(ep.dtype(), Type.Double);  // eigenvalues are real even for a complex-Hermitian input
    expect_unitary(Vp, "_aux_L", tol);
    expect_same_diagonal(ep, linalg::Eigh(M)[0], tol);
  }

  // TODO: issue #782: test reconstruction of M from U^-1 E U once Eig/Eigh return the inverse
  // eigenvector tensor U^-1 (which equals U^dagger only in the unitary/Hermitian case, and in
  // general carries bonds inherited from the column space of the input tensor).
}

/*=====test info=====
describe:Eig for fermionic tensors: complex spectrum and the eigen equation
  A V = V e (column-vector / row_v=false convention).
====================*/
TEST_F(linalg_Test, BkFUt_Eig) {
  const double tol = 1e-10;
  UniTensor A = make_square_fermionic({"i", "o"});
  A.at({0, 0}) = 1.0;
  A.at({0, 1}) = 2.0;
  A.at({1, 0}) = 0.5;
  A.at({1, 1}) = 1.3;
  A.at({2, 2}) = 2.0;
  A.at({2, 3}) = 0.4;
  A.at({3, 2}) = 0.9;
  A.at({3, 3}) = 3.0;

  auto out = linalg::Eig(A);
  ASSERT_EQ(out.size(), 2);
  UniTensor e = out[0], V = out[1];
  EXPECT_TRUE(e.is_diag());
  EXPECT_EQ(e.dtype(), Type.ComplexDouble);  // general (non-Hermitian) eigenvalues are complex

  // eigen equation: A V = V e. NOTE: this check relies on A having a SINGLE column leg ("o").
  // Relabeling V's row leg onto the column leg ("i" -> "o") and contracting then carries no
  // fermionic reorder sign. With two or more column legs (e.g. the rank-4 case below) the same
  // construction would have to swap an odd-parity column-leg pair, which picks up a sign that no
  // fermion_twists() placement cancels -- so the eigen equation is only verified here, and the
  // rank-4 sign-flip case is covered by spectrum invariance instead (full reconstruction: issue
  // #782).
  UniTensor Ac = A.astype(Type.ComplexDouble);
  UniTensor Vo = V.clone();
  Vo.relabel_("i", "o");
  UniTensor AV = Contract(Ac.fermion_twists(), Vo);
  UniTensor Ve = Contract(V, e);
  Ve.relabel_("_aux_R", "_aux_L");
  AV.permute_(std::vector<std::string>{"i", "_aux_L"});
  Ve.permute_(std::vector<std::string>{"i", "_aux_L"});
  EXPECT_TRUE((AV.apply() - Ve.apply()).Norm().item() < tol);

  // TODO: issue #782: test reconstruction of M from U^-1 E U once Eig/Eigh return the inverse
  // eigenvector tensor U^-1 (which equals U^dagger only in the unitary/Hermitian case, and in
  // general carries bonds inherited from the column space of the input tensor).

  // sign-flip coverage: a consistent permute of a (Hermitian, hence real-spectrum) operator
  // leaves the eigenvalues invariant but makes the pending signflip non-trivial, exercising the
  // negation path in Eig_BlockFermionic_UT_internal.
  UniTensor M = make_rank4_hermitian();
  UniTensor Mp = permute_with_signflips(M);
  expect_same_diagonal(linalg::Eig(M)[0], linalg::Eig(Mp)[0], tol);
}

/*=====test info=====
describe:Eig for a fermionic operator with genuinely COMPLEX eigenvalues (non-Hermitian blocks):
  the eigenvalue UniTensor must keep ComplexDouble dtype and report the correct complex eigenvalues
  with non-zero imaginary parts.
====================*/
TEST_F(linalg_Test, BkFUt_EigComplexEigenvalues) {
  const double tol = 1e-10;
  // even block [[1,-1],[1,1]] -> eigenvalues 1 +/- i ; odd block [[2,-0.5],[2,2]] -> 2 +/- i.
  UniTensor A = make_square_fermionic({"i", "o"});
  A.at({0, 0}) = 1.0;
  A.at({0, 1}) = -1.0;
  A.at({1, 0}) = 1.0;
  A.at({1, 1}) = 1.0;
  A.at({2, 2}) = 2.0;
  A.at({2, 3}) = -0.5;
  A.at({3, 2}) = 2.0;
  A.at({3, 3}) = 2.0;

  UniTensor e = linalg::Eig(A)[0];
  EXPECT_EQ(e.dtype(), Type.ComplexDouble);  // dtype must survive the block assignment

  std::vector<cytnx_complex128> eigvals;
  for (auto &bk : e.get_blocks_()) {
    Storage st = bk.clone().contiguous().storage();
    for (cytnx_uint64 i = 0; i < st.size(); i++) eigvals.push_back(st.at<cytnx_complex128>(i));
  }
  ASSERT_EQ(eigvals.size(), 4u);

  // the eigenvalues are genuinely complex (non-zero imaginary part)
  double max_imag = 0.0;
  for (auto z : eigvals) max_imag = std::max(max_imag, std::abs(z.imag()));
  EXPECT_GT(max_imag, 0.5);

  // each reference eigenvalue {1+i, 1-i, 2+i, 2-i} is matched by a returned eigenvalue
  std::vector<cytnx_complex128> ref = {{1, 1}, {1, -1}, {2, 1}, {2, -1}};
  for (auto r : ref) {
    double best = 1e18;
    for (auto z : eigvals) best = std::min(best, std::abs(z - r));
    EXPECT_TRUE(best < tol) << "missing eigenvalue " << r;
  }
}

/*=====test info=====
describe:Qr for fermionic tensors: reconstruction Q R = A, including a rank-4
  case with non-trivial fermionic sign flips.
====================*/
TEST_F(linalg_Test, BkFUt_Qr) {
  const double tol = 1e-10;
  UniTensor A = make_square_fermionic({"i", "o"});
  A.at({0, 0}) = 1.0;
  A.at({0, 1}) = 2.0;
  A.at({1, 0}) = 0.5;
  A.at({1, 1}) = 1.3;
  A.at({2, 2}) = 2.0;
  A.at({2, 3}) = 0.4;
  A.at({3, 2}) = 0.9;
  A.at({3, 3}) = 3.0;

  auto out = linalg::Qr(A);
  ASSERT_EQ(out.size(), 2);
  expect_unitary(out[0], "_aux_", tol);  // Q is an isometry: Q^dagger Q = I
  UniTensor recon = Contract(out[0], out[1]);
  recon.permute_(A.labels());
  EXPECT_TRUE((recon.apply() - A.apply()).Norm().item() < tol);

  // rank-4 case whose fermionic reordering produces non-trivial sign flips, run for a real and a
  // genuinely complex operator -- exercises the real (dgeqrf) and complex (zgeqrf) Qr code paths
  // plus the signflip negation in Qr_BlockFermionic_UT_internal. (Qr needs no Hermiticity.)
  for (bool cplx : {false, true}) {
    SCOPED_TRACE(cplx ? "ComplexDouble" : "Double");
    Bond ba = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 1}, {Symmetry::FermionParity()});
    Bond bb = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 1}, {Symmetry::FermionParity()});
    UniTensor T = UniTensor({ba, bb, ba.redirect(), bb.redirect()}, {"a", "b", "c", "d"}, 2,
                            cplx ? Type.ComplexDouble : Type.Double);
    cytnx_double val = 1.0;
    auto sh = T.shape();
    for (cytnx_uint64 i = 0; i < sh[0]; i++)
      for (cytnx_uint64 j = 0; j < sh[1]; j++)
        for (cytnx_uint64 k = 0; k < sh[2]; k++)
          for (cytnx_uint64 l = 0; l < sh[3]; l++) {
            auto proxy = T.at({i, j, k, l});
            if (proxy.exists()) {
              if (cplx)
                proxy = cytnx_complex128(val + 0.3, 0.2 * val - 0.1);
              else
                proxy = val + 0.3;
              val += 1.0;
            }
          }
    UniTensor Tp = T.permute({1, 0, 2, 3}).contiguous();
    bool anyflip = false;
    for (auto f : Tp.signflip()) anyflip = anyflip || f;
    EXPECT_TRUE(anyflip);  // guard: ensure the sign-flip code path is actually exercised

    auto out4 = linalg::Qr(Tp);
    ASSERT_EQ(out4.size(), 2);
    expect_unitary(out4[0], "_aux_", tol);  // Q is an isometry even with sign flips active
    UniTensor recon4 = Contract(out4[0], out4[1]);
    recon4.permute_(Tp.labels());
    EXPECT_TRUE((recon4.apply_() - Tp.apply_()).Norm().item() < tol);
  }
}

/*=====test info=====
describe:Eig/Eigh argument guards for fermionic tensors: row_v=true is rejected
  only when eigenvectors are requested (is_V), and rowrank must be < rank.
====================*/
TEST_F(linalg_Test, BkFUt_EigEighRowVGuards) {
  UniTensor H = make_square_fermionic({"i", "o"});
  H.at({0, 0}) = 2.0;
  H.at({0, 1}) = 0.7;
  H.at({1, 0}) = 0.7;
  H.at({1, 1}) = 3.0;
  H.at({2, 2}) = 1.5;
  H.at({2, 3}) = -0.4;
  H.at({3, 2}) = -0.4;
  H.at({3, 3}) = 2.5;

  // row_v=true is unsupported, but only when eigenvectors are requested.
  EXPECT_THROW({ linalg::Eig(H, true, true); }, std::logic_error);
  EXPECT_THROW({ linalg::Eigh(H, true, true); }, std::logic_error);
  // eigenvalue-only calls must not be rejected by the row_v flag.
  EXPECT_NO_THROW({ linalg::Eig(H, false, true); });
  EXPECT_NO_THROW({ linalg::Eigh(H, false, true); });

  // rowrank must be strictly less than rank.
  UniTensor sq = make_square_fermionic({"i", "o"});
  sq.set_rowrank_(2);
  EXPECT_THROW({ linalg::Eig(sq); }, std::logic_error);
  EXPECT_THROW({ linalg::Eigh(sq); }, std::logic_error);
}

/*=====test info=====
describe:test Gesvd_truncate (?gesvd-based SVD) unitarity and reconstruction for
  fermionic tensors with mixed in/out legs. Mirrors SvdUnitaryAndReconstruction.
====================*/
TEST_F(linalg_Test, BkFUt_GesvdTruncateUnitaryAndReconstruction) {
  const double tol = 1e-10;
  UniTensor M = make_mixed_inout_fermionic();
  // mixed in/out legs (no pending signflip)
  auto gesvd = linalg::Gesvd_truncate(M, 1000, 0., true, true, 0);
  expect_svd_reconstructs(gesvd, M, tol);
  // cross-check against Svd: ?gesvd and ?gesdd must yield the same singular values.
  expect_same_diagonal(gesvd[0], linalg::Svd_truncate(M, 1000, 0., true, 0)[0], tol);

  // sign-flip-active: a consistent permute makes the pending signflip non-trivial, exercising the
  // negation path in the BlockFermionic truncation while leaving U, S, vT correct. Run for both a
  // real and a genuinely complex-Hermitian operator so the real (?gesdd/?gesvd) and complex
  // (z-variant) code paths are both covered.
  for (bool cplx : {false, true}) {
    SCOPED_TRACE(cplx ? "ComplexDouble" : "Double");
    UniTensor Msf = permute_with_signflips(make_rank4_herm_dt(cplx));
    auto gesvd_sf = linalg::Gesvd_truncate(Msf, 1000, 0., true, true, 0);
    expect_svd_reconstructs(gesvd_sf, Msf, tol);
    expect_same_diagonal(gesvd_sf[0], linalg::Svd_truncate(Msf, 1000, 0., true, 0)[0], tol);
  }
}

/*=====test info=====
describe:ExpH for fermionic tensors: exp of a (sign-flip-active) Hermitian operator has spectrum
  exp(a * eigenvalues), verifying the per-sector signflip negation in ExpH_BlockUT_internal.
====================*/
TEST_F(linalg_Test, BkFUt_ExpH) {
  const double tol = 1e-10;
  const double a = 0.5;
  // sign-flip-active Hermitian operator (consistent permute keeps it Hermitian); run real and
  // genuinely complex-Hermitian so both the real and complex Eigh/ExpH code paths are covered.
  for (bool cplx : {false, true}) {
    SCOPED_TRACE(cplx ? "ComplexDouble" : "Double");
    UniTensor Mp = permute_with_signflips(make_rank4_herm_dt(cplx));
    UniTensor eM = linalg::ExpH(Mp, a);
    EXPECT_EQ(eM.uten_type(), UTenType.BlockFermionic);
    auto de = linalg::Eigh(eM);  // [eigenvalues, eigenvectors] of exp(a H)
    auto dm = linalg::Eigh(Mp);  // [eigenvalues, eigenvectors] of H
    // eigenvalues of exp(a H) are exp(a * eig(H)).
    expect_exp_spectrum(de[0], dm[0], a, tol);
    // exp(a H) shares H's eigenvectors: the two eigenbases agree up to a per-vector phase.
    expect_same_eigenvectors(de[1], dm[1], "_aux_L", tol);
  }
}

/*=====test info=====
describe:ExpM for fermionic tensors: exp of a (sign-flip-active) operator has spectrum
  exp(a * eigenvalues), verifying the per-sector signflip negation in ExpM_BlockUT_internal.
====================*/
TEST_F(linalg_Test, BkFUt_ExpM) {
  const double tol = 1e-10;
  const double a = 0.3;
  // ExpM accepts a general operator; a Hermitian one gives a real spectrum so the sorted compare
  // is well defined, while the consistent permute still exercises the signflip negation path.
  UniTensor Mp = permute_with_signflips(make_rank4_hermitian());
  UniTensor eM = linalg::ExpM(Mp, a);
  EXPECT_EQ(eM.uten_type(), UTenType.BlockFermionic);
  // both operands are Hermitian, so use Eigh (sorted, orthonormal) for a well-defined comparison
  // even though ExpM itself uses the general Eig internally.
  auto de = linalg::Eigh(eM);  // [eigenvalues, eigenvectors] of exp(a M)
  auto dm = linalg::Eigh(Mp);  // [eigenvalues, eigenvectors] of M
  expect_exp_spectrum(de[0], dm[0], a, tol);
  expect_same_eigenvectors(de[1], dm[1], "_aux_L", tol);
}

/*=====test info=====
describe:ExpM bias term for fermionic tensors: ExpM(M, a=0, b) must equal exp(0*M + b*I) = exp(b)*I,
  i.e. the bias b must NOT be dropped by the a==0 fast path.
====================*/
TEST_F(linalg_Test, BkFUt_ExpMBias) {
  const double tol = 1e-10;
  const double b = 0.5;
  UniTensor Mp = permute_with_signflips(make_rank4_hermitian());  // sign-flip-active
  UniTensor eM0 = linalg::ExpM(Mp, 0.0, 0.0);  // exp(0) = I
  UniTensor eMb = linalg::ExpM(Mp, 0.0, b);  // exp(0*M + b*I) = exp(b) * I
  EXPECT_EQ(eMb.uten_type(), UTenType.BlockFermionic);
  // the bias is preserved: exp(0,b) = exp(b) * exp(0,0). (With the bug eMb == eM0 != exp(b)*eM0.)
  EXPECT_TRUE((eMb - eM0 * std::exp(b)).Norm().item() < tol);
  // and exp(0,0) is genuinely the identity (all eigenvalues 1).
  for (double x : sorted_diagonal(linalg::Eigh(eM0)[0])) EXPECT_NEAR(x, 1.0, tol);
}

// Helper: rank-5 fermionic UniTensor (legs a,b,c,d,e; rowrank 3) populated with sequential values
// 1..8 (or their reciprocals when `inverse` is set) over its eight existing charge-0 components.
inline UniTensor make_rank5_fermionic_seq(bool inverse) {
  Bond B1 = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 1}, {Symmetry::FermionParity()});
  Bond B2 = Bond(BD_IN, {Qs(0), Qs(1)}, {1, 1}, {Symmetry::FermionParity()});
  Bond B12 = B1.combineBond(B2).redirect_();
  Bond B3 = Bond(BD_OUT, {Qs(0) >> 1, Qs(1) >> 1}, {Symmetry::FermionParity()});
  Bond B4 = Bond(BD_IN, {Qs(0) >> 1, Qs(1) >> 1}, {Symmetry::FermionParity()});
  UniTensor M = UniTensor({B1, B2, B12, B3, B4}, {"a", "b", "c", "d", "e"});
  M.at({0, 0, 0, 0, 0}) = inverse ? 1. / 1. : 1.;
  M.at({0, 0, 1, 0, 0}) = inverse ? 1. / 2. : 2.;
  M.at({0, 1, 2, 0, 0}) = inverse ? 1. / 3. : 3.;
  M.at({0, 1, 3, 0, 0}) = inverse ? 1. / 4. : 4.;
  M.at({1, 0, 2, 0, 0}) = inverse ? 1. / 5. : 5.;
  M.at({1, 0, 3, 0, 0}) = inverse ? 1. / 6. : 6.;
  M.at({1, 1, 0, 0, 0}) = inverse ? 1. / 7. : 7.;
  M.at({1, 1, 1, 0, 0}) = inverse ? 1. / 8. : 8.;
  return M;
}

/*=====test info=====
describe:test pseudo-inverse
====================*/
TEST_F(linalg_Test, BkFUt_Inv) {
  const double tol = 1e-14;
  double clip = 1e-15;
  UniTensor BFUT3 = make_rank5_fermionic_seq(false);
  UniTensor BFUT3INV = make_rank5_fermionic_seq(true);
  EXPECT_TRUE(AreNearlyEqUniTensor(BFUT3.Inv(clip), BFUT3INV, tol));
  UniTensor T = BFUT3.permute({3, 1, 4, 2, 0}).contiguous();
  EXPECT_TRUE(AreNearlyEqUniTensor(T.Inv(clip).Inv_(clip), T, tol));
  EXPECT_FALSE(AreNearlyEqUniTensor(T.Inv(clip), T, tol));
  clip = 0.1;  // test actual clipping as well
  auto tmp = T.clone();
  tmp.Inv_(clip);  // test inline version
  EXPECT_TRUE(AreEqUniTensor(T.Inv(clip), tmp));
  tmp = T.clone();
  for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++)
      for (size_t k = 0; k < 4; k++)
        for (size_t l = 0; l < 2; l++)
          for (size_t m = 0; m < 2; m++) {
            // elements are permuted!
            auto proxy = tmp.at({l, j, m, k, i});
            if (proxy.exists()) {
              Scalar val = proxy;
              if (val.abs() <= clip)
                proxy = 0.;
              else
                proxy = 1. / proxy;
            }
          }
  EXPECT_TRUE(AreNearlyEqUniTensor(T.Inv(clip), tmp, tol));
}

/*=====test info=====
describe:test power
====================*/
TEST_F(linalg_Test, BkFUt_Pow) {
  const double tol = 1e-14;
  UniTensor BFUT3 = make_rank5_fermionic_seq(false);
  UniTensor T = BFUT3.permute({3, 1, 4, 2, 0}).contiguous();
  // only even powers are products, odd powers differ by signflips
  EXPECT_TRUE(AreNearlyEqUniTensor(T.Pow(3.), T * T * T, tol));
  auto tmp = T.clone();
  tmp.Pow_(2.3);  // test inline version
  EXPECT_TRUE(AreEqUniTensor(T.Pow(2.3), tmp));
  for (double p = 0.; p < 1.6; p += 0.5) {
    tmp = T.clone();
    for (size_t i = 0; i < 2; i++)
      for (size_t j = 0; j < 2; j++)
        for (size_t k = 0; k < 4; k++)
          for (size_t l = 0; l < 2; l++)
            for (size_t m = 0; m < 2; m++) {
              // elements are permuted!
              auto proxy = tmp.at({l, j, m, k, i});
              if (proxy.exists()) {
                Scalar val = proxy;
                proxy = std::pow((cytnx_double)val, p);
              }
            }
    EXPECT_TRUE(AreNearlyEqUniTensor(T.Pow(p), tmp, tol));
  }
}
