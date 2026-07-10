#include "cytnx.hpp"
#include <gtest/gtest.h>
#include "../test_tools.h"

using namespace cytnx;
using namespace testing;
using namespace TestTools;

namespace SvdTruncateTest {

  static TestFailMsg fail_msg;

  bool CheckResult(const std::string& case_name, const cytnx_uint64& keepdim);
  bool CheckResult(const std::string& case_name, const cytnx_uint64& keepdim,
                   const std::vector<cytnx_uint64> min_blockdim);
  bool ReComposeCheck(const UniTensor& Tin, const std::vector<UniTensor>& Tout);
  bool CheckLabels(const UniTensor& Tin, const std::vector<UniTensor>& Tout);
  bool SingularValsCorrect(const UniTensor& res, const UniTensor& ans);
  std::string src_data_root = CYTNX_TEST_DATA_DIR "/common/";
  std::string ans_data_root = CYTNX_TEST_DATA_DIR "/linalg/Svd_truncate/";
  // normal test

  /*=====test info=====
  describe:Test dense UniTensor only one element.
  input:
    T:Dense UniTensor only one element.
    is_U:true
    is_VT:true
  ====================*/
  TEST(Svd_truncate, dense_one_elem) {
    std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
    fail_msg.Init(test_case_name);
    int size = 1;
    std::vector<Bond> bonds = {Bond(size), Bond(size), Bond(size)};
    int rowrank = 1;
    bool is_diag = false;
    auto labels = std::vector<std::string>();
    auto T = UniTensor(bonds, labels, rowrank, cytnx::Type.Double, cytnx::Device.cpu, is_diag);
    random::uniform_(T, -10, 0, 0);
    std::vector<UniTensor> Svds = linalg::Svd_truncate(T, 1);
    EXPECT_TRUE(CheckLabels(T, Svds)) << fail_msg.TraceFailMsgs();
    EXPECT_TRUE(ReComposeCheck(T, Svds)) << fail_msg.TraceFailMsgs();
    EXPECT_EQ(Svds[0].at<double>({0}), std::abs(T.at<double>({0, 0, 0})))
      << "Singular value is wrong."
      << " line:" << __LINE__ << std::endl;
  }

  // /*=====test info=====
  // describe:Test Dense UniTensor.
  // input:
  //   T:Dense UniTensor with real or complex real type.
  //   is_U:true
  //   is_VT:true
  // ====================*/
  // TEST(Svd_truncate, dense_nondiag_test) {
  //   std::vector<std::string> case_list = {"dense_nondiag_C128", "dense_nondiag_F64"};
  //   for (const auto& case_name : case_list) {
  //     std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
  //     fail_msg.Init(test_case_name + ", " + case_name);
  //     EXPECT_TRUE(CheckResult(case_name, 4)) << fail_msg.TraceFailMsgs();
  //   }
  // }

  // error test

  /*=====test info=====
  describe:error test, Test Dense diagonal tensor.
  input:
    T:Dense diagonal complex real type UniTensor.
    is_U:true
    is_VT:true
  ====================*/
  TEST(Svd_truncate, err_dense_diag_test) {
    int size = 5;
    std::vector<Bond> bonds = {Bond(size), Bond(size)};
    int rowrank = 1;
    bool is_diag = true;
    auto labels = std::vector<std::string>();
    auto T = UniTensor(bonds, labels, rowrank, cytnx::Type.Double, cytnx::Device.cpu, is_diag);
    random::uniform_(T, 0, 10, 0);
    EXPECT_THROW({ std::vector<UniTensor> Svds = linalg::Svd_truncate(T, 2); }, std::logic_error);
  }

  /*=====test info=====
  describe:Test Dense UniTensor with exponentially decaying singular values.
  input:
    T:Dense UniTensor with real or complex real type.
    is_U:true
    is_VT:true
  ====================*/
  TEST(Svd_truncate, dense_exp_svals_test) {
    std::vector<std::string> case_list = {"dense_nondiag_exp_Svals_C128",
                                          "dense_nondiag_exp_Svals_F64"};
    for (const auto& case_name : case_list) {
      std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
      fail_msg.Init(test_case_name + ", " + case_name);
      EXPECT_TRUE(CheckResult(case_name, 5)) << fail_msg.TraceFailMsgs();
    }
  }

  /*=====test info=====
  describe:Test U(1) UniTensor with exponentially decaying singular values and use of min_blockdim;
  input:
    T:U(1) UniTensor with real or complex real type.
    is_U:true
    is_VT:true
  ====================*/
  TEST(Svd_truncate, U1_exp_svals_minblockdim_test) {
    std::vector<std::string> case_list = {"sym_UT_U1_exp_Svals_C128", "sym_UT_U1_exp_Svals_F64"};
    for (const auto& case_name : case_list) {
      std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
      fail_msg.Init(test_case_name + ", " + case_name);
      EXPECT_TRUE(CheckResult(case_name, 5, {1, 1, 0, 2, 0})) << fail_msg.TraceFailMsgs();
    }
  }

  bool ReComposeCheck(const UniTensor& Tin, const std::vector<UniTensor>& Tout) {
    bool is_double_float_acc = true;
    auto dtype = Tin.dtype();
    if (dtype == Type.Float || dtype == Type.ComplexFloat) {
      is_double_float_acc = false;
    }
    const UniTensor& S = Tout[0];
    const UniTensor& U = Tout[1];
    const UniTensor& V = Tout[2];
    UniTensor ReCompose = Contract(U, S);
    ReCompose = Contract(ReCompose, V);
    const double tol = is_double_float_acc ? 1.0e-14 : 1.0e-6;
    auto T_float = Tin.clone();
    if (Tin.dtype() > Type.Float) {
      T_float = Tin.astype(Type.Double);
    }
    bool is_eq = AreNearlyEqUniTensor(T_float, ReCompose.permute_(T_float.labels()), tol);
    return is_eq;
  }

  bool CheckLabels(const UniTensor& Tin, const std::vector<UniTensor>& Tout) {
    const std::vector<std::string>& in_labels = Tin.labels();
    const std::vector<std::string>& s_labels = Tout[0].labels();
    const std::vector<std::string>& u_labels = Tout[1].labels();
    const std::vector<std::string>& v_labels = Tout[2].labels();
    // check S
    if (s_labels[0] != "_aux_L") {
      fail_msg.AppendMsg("The label of the left leg in 'S' is wrong. ", __func__, __LINE__);
      return false;
    }
    if (s_labels[1] != "_aux_R") {
      fail_msg.AppendMsg("The label of the left leg in 'S' is wrong. ", __func__, __LINE__);
      return false;
    }
    // check U
    for (size_t i = 0; i < u_labels.size() - 1; ++i) {  // exclude U final lags label
      if (u_labels[i] != in_labels[i]) {
        fail_msg.AppendMsg("The label of 'U' is wrong. ", __func__, __LINE__);
        return false;
      }
    }
    // check V
    for (size_t i = 1; i < v_labels.size(); ++i) {  // exclude VT first lags label
      auto in_indx = u_labels.size() - 2 + i;
      if (v_labels[i] != in_labels[in_indx]) {
        fail_msg.AppendMsg("The label of 'V' is wrong. ", __func__, __LINE__);
        return false;
      }
    }
    return true;
  }

  bool SingularValsCorrect(const UniTensor& res, const UniTensor& ans) {
    bool is_double_float_acc = true;
    auto dtype = res.dtype();
    if (dtype == Type.Float || dtype == Type.ComplexFloat) {
      is_double_float_acc = false;
    }
    // relative error = |ans-res| / x
    //   x = |ans| < 1.0 ? 1.0 : x
    Tensor diff_tens = (ans - res).Norm();
    double ans_norm = (ans.Norm().storage()).at<double>(0);
    ans_norm = ans_norm < 1.0 ? 1.0 : ans_norm;
    double relative_err = (diff_tens.storage()).at<double>(0) / ans_norm;

    const double tol = is_double_float_acc ? 1.0e-14 : 1.0e-6;
    return (relative_err < tol);
  }

  // no use
  void Check_UU_VV_Identity(const UniTensor& Tin, const std::vector<UniTensor>& Tout) {
    const UniTensor& U = Tout[1];
    const UniTensor& V = Tout[2];
    auto UD = U.Dagger();
    UD.relabel_({"0", "1", "9"});
    UD.permute_({2, 0, 1}, 1);
    auto UUD = Contract(U, UD);
  }

  bool CheckResult(const std::string& case_name, const cytnx_uint64& keepdim) {
    // test data source file
    std::string src_file_name = src_data_root + case_name + ".cytnx";
    // anscer file
    std::string ans_file_name = ans_data_root + case_name + ".cytnx";
    // reconstructed matrix file
    std::string rec_file_name = ans_data_root + case_name + "_reconstructed.cytnx";
    // bool need_U, need_VT;
    bool compute_uv;
    UniTensor src_T = UniTensor::Load(src_file_name);
    UniTensor ans_T = UniTensor::Load(ans_file_name);  // singular values UniTensor
    UniTensor rec_T = UniTensor::Load(rec_file_name);  // M = U * S * V after correct truncated SVD

    // Do Svd_truncate
    std::vector<UniTensor> Svds = linalg::Svd_truncate(src_T, keepdim, 0., true, 0);

    // check labels
    if (!(CheckLabels(src_T, Svds))) {
      fail_msg.AppendMsg("The output labels are wrong. ", __func__, __LINE__);
      return false;
    }

    // check answer
    if (!(SingularValsCorrect(Svds[0], ans_T))) {
      fail_msg.AppendMsg("The singular values are wrong. ", __func__, __LINE__);
      return false;
    }

    // check recompose [M - USV*]
    if (!ReComposeCheck(rec_T, Svds)) {
      fail_msg.AppendMsg(
        "The result is wrong after recomposing. "
        "That's mean T not equal USV* ",
        __func__, __LINE__);
      return false;
    }

    return true;
  }

  bool CheckResult(const std::string& case_name, const cytnx_uint64& keepdim,
                   const std::vector<cytnx_uint64> min_blockdim) {
    // test data source file
    std::string src_file_name = src_data_root + case_name + ".cytnx";
    // anscer file
    std::string ans_file_name = ans_data_root + case_name + "_minblockdim.cytnx";
    // reconstructed matrix file
    std::string rec_file_name = ans_data_root + case_name + "_minblockdim_reconstructed.cytnx";
    // bool need_U, need_VT;
    bool compute_uv;
    UniTensor src_T = UniTensor::Load(src_file_name);
    UniTensor ans_T = UniTensor::Load(ans_file_name);  // singular values UniTensor
    UniTensor rec_T = UniTensor::Load(rec_file_name);  // M = U * S * V after correct truncated SVD

    // Do Svd_truncate
    std::vector<UniTensor> Svds = linalg::Svd_truncate(src_T, keepdim, min_blockdim, 0., true, 0);

    // check labels
    if (!(CheckLabels(src_T, Svds))) {
      fail_msg.AppendMsg("The output labels are wrong. ", __func__, __LINE__);
      return false;
    }

    ans_T.print_diagram();
    Svds[0].print_diagram();
    // check answer
    if (!(SingularValsCorrect(Svds[0], ans_T))) {
      fail_msg.AppendMsg("The singular values are wrong. ", __func__, __LINE__);
      return false;
    }

    // check recompose [M - USV*]
    if (!ReComposeCheck(rec_T, Svds)) {
      fail_msg.AppendMsg(
        "The result is wrong after recomposing. "
        "That's mean T not equal USV* ",
        __func__, __LINE__);
      return false;
    }

    return true;
  }

  /*=====test info=====
  describe:Tensor-level Svd_truncate must work for both is_UvT=true and is_UvT=false (the latter
  previously read out-of-range vector slots) and every return_err mode. Singular-value content is
  validated against a full Svd reference via CheckTruncatedSvdResult; U/vT shapes are checked
  inline (their gauge depends on LAPACK gesdd mode so we don't compare values).
  ====================*/
  TEST(Svd_truncate, flag_combinations_dense) {
    for (auto dtype : {Type.Double, Type.ComplexDouble}) {
      Tensor T = Tensor({6, 5}, dtype);
      InitTensorUniform(T, /*seed=*/1);
      const cytnx_uint64 keep = 3;

      // full reference: untruncated Svd provides all min(6,5)=5 singular values
      std::vector<Tensor> full_ref = linalg::Svd(T, /*is_UvT=*/true);

      for (bool is_UvT : {false, true}) {
        for (int return_err : {0, 1, 2}) {
          std::vector<Tensor> out = linalg::Svd_truncate(T, keep, 0., is_UvT, return_err, 1);
          const std::string label =
            "is_UvT=" + std::to_string(is_UvT) + " return_err=" + std::to_string(return_err);
          // Svd packs is_UvT as both U and vT
          CheckTruncatedSvdResult(out, full_ref[0], keep, is_UvT, is_UvT, return_err, 1e-12, label);

          if (is_UvT) {
            EXPECT_EQ(out[1].shape(), std::vector<cytnx_uint64>({6, keep})) << label;
            EXPECT_EQ(out[2].shape(), std::vector<cytnx_uint64>({keep, 5})) << label;
          }
        }
      }
    }
  }

  /*=====test info=====
  describe:When keepdim >= #singular values and err=0, nothing is truncated; the returned error
  tensor must be a 1-element zero (the truncation error is zero).
  ====================*/
  TEST(Svd_truncate, no_truncation_returns_zero_error) {
    Tensor T = Tensor({6, 5}, Type.Double);
    InitTensorUniform(T, 7);
    const cytnx_uint64 full = 5;  // min(6, 5) singular values
    std::vector<Tensor> full_ref = linalg::Svd(T, /*is_UvT=*/true);
    for (int return_err : {1, 2}) {
      std::vector<Tensor> out =
        linalg::Svd_truncate(T, /*keepdim=*/full, 0., /*is_UvT=*/true, return_err, 1);
      const std::string label = "return_err=" + std::to_string(return_err);
      // is_UvT=true and keep == full means [S, U, vT, terr] with a 1-element zero terr
      CheckTruncatedSvdResult(out, full_ref[0], full, true, true, return_err, 1e-12, label);
      EXPECT_EQ(out[1].shape(), std::vector<cytnx_uint64>({6, full})) << label;
      EXPECT_EQ(out[2].shape(), std::vector<cytnx_uint64>({full, 5})) << label;
    }
  }

  /*=====test info=====
  describe:When per-block min_blockdim guarantees already meet or exceed the global keepdim cap
  (keep_dim<=0 internally) but there are further singular values, the return_err path must report
  the actually dropped singular values, not a 1-element zero tensor.
  ====================*/
  TEST(Svd_truncate, return_err_when_min_blockdim_exhausts_keepdim) {
    // 2-leg U(1) BlockUT, 3 sectors of size 3x3 each. With min_blockdim=[2,2,2] every block
    // keeps its top 2 SVs and contributes its 3rd to Sall (3 dropped values total);
    // keep_dim = 3 - 6 = -3 < 0, so all 3 are dropped via the keep_dim<=0 branch.
    auto syms = std::vector<Symmetry>{Symmetry(SymmetryType::U)};
    Bond bk = Bond(BD_KET, {{0}, {1}, {2}}, {3, 3, 3}, syms);
    UniTensor src_T({bk, bk.redirect()}, {"l", "r"}, 1, Type.Double, Device.cpu, false);
    InitUniTensorUniform(src_T, 23);

    const cytnx_uint64 keepdim = 3;
    const std::vector<cytnx_uint64> min_blockdim = {2, 2, 2};

    // return_err = 2: terr holds every dropped singular value (3 total)
    std::vector<UniTensor> svds_all =
      linalg::Svd_truncate(src_T, keepdim, min_blockdim, 0., true, 2, 1);
    Tensor terr_all = svds_all.back().get_block_();
    ASSERT_EQ(terr_all.shape(), std::vector<cytnx_uint64>({3}))
      << "terr must hold all 3 dropped singular values, not a 1-element zero";
    double max_abs = 0.0;
    for (cytnx_uint64 i = 0; i < terr_all.storage().size(); ++i)
      max_abs = std::max(max_abs, std::abs(terr_all.storage().at<double>(i)));
    EXPECT_GT(max_abs, 0.0) << "terr values are all zero (previous bug)";

    // return_err = 1: terr is a scalar = largest dropped singular value
    std::vector<UniTensor> svds_max =
      linalg::Svd_truncate(src_T, keepdim, min_blockdim, 0., true, 1, 1);
    Tensor terr_max = svds_max.back().get_block_();
    ASSERT_EQ(terr_max.shape(), std::vector<cytnx_uint64>{});
    EXPECT_NEAR(std::abs(terr_max.storage().at<double>(0)), max_abs, 1e-10 * (1.0 + max_abs));
  }

  /*=====test info=====
  describe:Setting min_blockdim must keep at least the requested number of singular values per
  sector. With a small keepdim, the baseline (no min_blockdim) drops entire sectors; with
  min_blockdim={1,1,1} every sector keeps at least one singular value.
  ====================*/
  TEST(Svd_truncate, min_blockdim_keeps_each_sector) {
    auto syms = std::vector<Symmetry>{Symmetry(SymmetryType::U)};
    Bond bk = Bond(BD_KET, {{0}, {1}, {2}}, {3, 3, 3}, syms);
    UniTensor src_T({bk, bk.redirect()}, {"l", "r"}, 1, Type.Double, Device.cpu, false);
    InitUniTensorUniform(src_T, 29);
    const cytnx_uint64 keepdim = 1;

    // baseline: only the single largest singular value is kept, so most sectors disappear
    std::vector<UniTensor> base = linalg::Svd_truncate(src_T, keepdim);
    const auto base_degs = base[0].bonds()[0].getDegeneracies();
    EXPECT_LT(base_degs.size(), 3u)
      << "baseline should drop some sectors; pick a smaller keepdim or larger blocks";

    // with min_blockdim={1,1,1}: every sector must keep at least 1
    std::vector<UniTensor> withf = linalg::Svd_truncate(src_T, keepdim, {1, 1, 1});
    const auto withf_degs = withf[0].bonds()[0].getDegeneracies();
    ASSERT_EQ(withf_degs.size(), 3u);
    for (cytnx_uint64 b = 0; b < 3; ++b) EXPECT_GE(withf_degs[b], 1u) << "block " << b;
  }

  /*=====test info=====
  describe:With err larger than every singular value, the truncation loop would otherwise cut
  down to 1. mindim must restrict the kept count to at least mindim, and the kept values must be
  the mindim largest ones of the full SVD.
  ====================*/
  TEST(Svd_truncate, mindim_floor) {
    Tensor T = Tensor({6, 5}, Type.Double);
    InitTensorUniform(T, 13);
    const cytnx_uint64 full = 5;
    const cytnx_uint64 floor = 3;
    std::vector<Tensor> ref = linalg::Svd(T, /*is_UvT=*/true);

    std::vector<Tensor> out =
      linalg::Svd_truncate(T, /*keepdim=*/full, /*err=*/1e9, /*is_UvT=*/true, 0, floor);
    ASSERT_EQ(out[0].shape(), std::vector<cytnx_uint64>({floor}));
    for (cytnx_uint64 i = 0; i < floor; ++i) {
      const double got = out[0].astype(Type.Double).storage().at<double>(i);
      const double exp = ref[0].astype(Type.Double).storage().at<double>(i);
      EXPECT_NEAR(got, exp, 1e-12 * (1.0 + std::abs(exp))) << "S[" << i << "]";
    }
  }
}  // namespace SvdTruncateTest
