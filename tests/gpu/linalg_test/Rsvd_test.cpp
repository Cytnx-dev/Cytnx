#include <gtest/gtest.h>

#include "cytnx.hpp"
#include "gpu_test_tools.h"

using namespace cytnx;
using namespace testing;
using namespace TestTools;

namespace RsvdTest {

  static TestFailMsg fail_msg;

  bool CheckResult(const std::string& case_name, const cytnx_uint64& keepdim,
                   const cytnx_uint64& power_iteration);
  bool CheckResult(const std::string& case_name, const cytnx_uint64& keepdim,
                   const std::vector<cytnx_uint64> min_blockdim,
                   const cytnx_uint64& power_iteration);
  bool ReComposeCheck(const UniTensor& Tin, const std::vector<UniTensor>& Tout);
  bool CheckLabels(const UniTensor& Tin, const std::vector<UniTensor>& Tout);
  bool SingularValsCorrect(const UniTensor& res, const UniTensor& ans);
  UniTensor BuildCombinedBlockFermionicTensorWithSignflip();
  UniTensor BuildLowRankRectangularDenseUniTensor(const int device);
  void CheckLowRankRectangularDenseUniTensorCase(const UniTensor& src_T, const UniTensor& src_Tt);
  std::string src_data_root = CYTNX_TEST_DATA_DIR "/common/";
  std::string ans_data_root = CYTNX_TEST_DATA_DIR "/linalg/Svd_truncate/";

  /*=====test info=====
  describe:Test dense UniTensor only one element.
  input:
    T:Dense UniTensor only one element.
    is_U:true
    is_VT:true
  ====================*/
  TEST(Rsvd, gpu_dense_one_elem) {
    std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
    fail_msg.Init(test_case_name);
    int size = 1;
    std::vector<Bond> bonds = {Bond(size), Bond(size), Bond(size)};
    int rowrank = 1;
    bool is_diag = false;
    auto labels = std::vector<std::string>();
    auto T = UniTensor(bonds, labels, rowrank, cytnx::Type.Double, cytnx::Device.cuda, is_diag);
    random::Make_uniform(T, -10, 0, 0);
    std::vector<UniTensor> rsvds = linalg::Rsvd(T, 1);
    EXPECT_TRUE(CheckLabels(T, rsvds)) << fail_msg.TraceFailMsgs();
    EXPECT_TRUE(ReComposeCheck(T, rsvds)) << fail_msg.TraceFailMsgs();
    EXPECT_DOUBLE_EQ(rsvds[0].at<double>({0}), std::abs(T.at<double>({0, 0, 0})))
      << "Singular value is wrong."
      << " line:" << __LINE__ << std::endl;
  }

  TEST(Rsvd, gpu_dense_low_rank_rectangular_and_transposed_exact_reconstruction) {
#ifndef UNI_CUQUANTUM
    GTEST_SKIP() << "GPU randomized SVD dense tests require cuQuantum-enabled GPU QR support.";
#endif
    std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
    fail_msg.Init(test_case_name);

    UniTensor src_T = BuildLowRankRectangularDenseUniTensor(Device.cuda);
    UniTensor src_Tt = src_T.permute({2, 0, 1}, 1).contiguous_();

    CheckLowRankRectangularDenseUniTensorCase(src_T, src_Tt);
  }

  /*=====test info=====
  describe:error test, Test Dense diagonal tensor.
  input:
    T:Dense diagonal complex real type UniTensor.
    is_U:true
    is_VT:true
  ====================*/
  TEST(Rsvd, gpu_err_dense_diag_test) {
    int size = 5;
    std::vector<Bond> bonds = {Bond(size), Bond(size)};
    int rowrank = 1;
    bool is_diag = true;
    auto labels = std::vector<std::string>();
    auto T = UniTensor(bonds, labels, rowrank, cytnx::Type.Double, cytnx::Device.cuda, is_diag);
    random::Make_uniform(T, 0, 10, 0);
    EXPECT_THROW({ std::vector<UniTensor> rsvds = linalg::Rsvd(T, 2); }, std::logic_error);
  }

  /*=====test info=====
  describe:Test Dense UniTensor with exponentially decaying singular values.
  input:
    T:Dense UniTensor with real or complex real type.
    is_U:true
    is_VT:true
  ====================*/
  TEST(Rsvd, gpu_dense_exp_svals_test) {
#ifndef UNI_CUQUANTUM
    GTEST_SKIP() << "GPU randomized SVD dense tests require cuQuantum-enabled GPU QR support.";
#endif
    std::vector<std::string> case_list = {"dense_nondiag_exp_Svals_C128",
                                          "dense_nondiag_exp_Svals_F64"};
    for (const auto& case_name : case_list) {
      std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
      fail_msg.Init(test_case_name + ", " + case_name);
      EXPECT_TRUE(CheckResult(case_name, 5, 2)) << fail_msg.TraceFailMsgs();
    }
  }

  /*=====test info=====
  describe:Test Dense UniTensor with exponentially decaying singular values. No power iteration in
  Rsvd. input: T:Dense UniTensor with real or complex real type. is_U:true is_VT:true
  ====================*/
  TEST(Rsvd, gpu_dense_exp_svals_no_power_iteration_test) {
#ifndef UNI_CUQUANTUM
    GTEST_SKIP() << "GPU randomized SVD dense tests require cuQuantum-enabled GPU QR support.";
#endif
    std::vector<std::string> case_list = {"dense_nondiag_exp_Svals_C128",
                                          "dense_nondiag_exp_Svals_F64"};
    for (const auto& case_name : case_list) {
      std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
      fail_msg.Init(test_case_name + ", " + case_name);
      EXPECT_TRUE(CheckResult(case_name, 5, 0)) << fail_msg.TraceFailMsgs();
    }
  }

  /*=====test info=====
  describe:Test U(1) UniTensor with exponentially decaying singular values.
  input:
    T:U(1) UniTensor with real or complex real type.
    is_U:true
    is_VT:true
  ====================*/
  TEST(Rsvd, gpu_U1_exp_svals_test) {
    std::vector<std::string> case_list = {"sym_UT_U1_exp_Svals_C128", "sym_UT_U1_exp_Svals_F64"};
    for (const auto& case_name : case_list) {
      std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
      fail_msg.Init(test_case_name + ", " + case_name);
      EXPECT_TRUE(CheckResult(case_name, 5, 2)) << fail_msg.TraceFailMsgs();
    }
  }

  /*=====test info=====
  describe:Test U(1) UniTensor with exponentially decaying singular values and use of min_blockdim;
  input:
    T:U(1) UniTensor with real or complex real type.
    is_U:true
    is_VT:true
  ====================*/
  TEST(Rsvd, gpu_U1_exp_svals_minblockdim_test) {
    std::vector<std::string> case_list = {"sym_UT_U1_exp_Svals_C128", "sym_UT_U1_exp_Svals_F64"};
    for (const auto& case_name : case_list) {
      std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
      fail_msg.Init(test_case_name + ", " + case_name);
      EXPECT_TRUE(CheckResult(case_name, 5, {1, 1, 0, 2, 0}, 2)) << fail_msg.TraceFailMsgs();
    }
  }

  /*=====test info=====
  describe:Test BlockFermionic UniTensor.
  input:
    T:BlockFermionic UniTensor.
    is_U:true
    is_VT:true
  ====================*/
  TEST(Rsvd, gpu_block_fermionic_test) {
    std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
    fail_msg.Init(test_case_name);
    UniTensor src_T = BuildCombinedBlockFermionicTensorWithSignflip();
    std::vector<UniTensor> rsvds = linalg::Rsvd(src_T, 10, 0., true, true, 0, 0, 2, 1, 2, 0);
    EXPECT_TRUE(CheckLabels(src_T, rsvds)) << fail_msg.TraceFailMsgs();
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
    UniTensor recomposed = Contract(U, S);
    recomposed = Contract(recomposed, V);
    const double tol = is_double_float_acc ? 1.0e-9 : 1.0e-2;
    auto T_float = Tin.clone();
    if (Tin.dtype() > Type.Float) {
      T_float = Tin.astype(Type.Double);
    }
    bool is_eq = AreNearlyEqUniTensor(T_float, recomposed.permute_(T_float.labels()), tol);
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
    Tensor diff_tens = (ans - res).Norm();
    double ans_norm = (ans.Norm().storage()).at<double>(0);
    ans_norm = ans_norm < 1.0 ? 1.0 : ans_norm;
    double relative_err = (diff_tens.storage()).at<double>(0) / ans_norm;
    const double tol = is_double_float_acc ? 1.0e-14 : 1.0e-6;
    return (relative_err < tol);
  }

  UniTensor BuildLowRankRectangularDenseUniTensor(const int device) {
    Tensor left = arange(0, 18, 1, Type.Double, device).reshape({6, 3}) + 1.0;
    Tensor right = arange(0, 15, 1, Type.Double, device).reshape({3, 5}) + 1.0;
    Tensor src = linalg::Matmul(left, right).reshape({2, 3, 5});
    return UniTensor(src, false, 2, {"a", "b", "c"});
  }

  void CheckLowRankRectangularDenseUniTensorCase(const UniTensor& src_T, const UniTensor& src_Tt) {
    const cytnx_uint64 keepdim = 3;
    std::vector<UniTensor> rsvd_src =
      linalg::Rsvd(src_T, keepdim, 0., true, true, 0, 1, 0, 0., 0, 7);
    std::vector<UniTensor> rsvd_src_t =
      linalg::Rsvd(src_Tt, keepdim, 0., true, true, 0, 1, 0, 0., 0, 7);

    ASSERT_EQ(rsvd_src.size(), 3UL);
    ASSERT_EQ(rsvd_src_t.size(), 3UL);

    EXPECT_TRUE(CheckLabels(src_T, rsvd_src)) << fail_msg.TraceFailMsgs();
    EXPECT_TRUE(CheckLabels(src_Tt, rsvd_src_t)) << fail_msg.TraceFailMsgs();
    EXPECT_TRUE(SingularValsCorrect(rsvd_src[0], rsvd_src_t[0])) << fail_msg.TraceFailMsgs();
    EXPECT_TRUE(ReComposeCheck(src_T, rsvd_src)) << fail_msg.TraceFailMsgs();
    EXPECT_TRUE(ReComposeCheck(src_Tt, rsvd_src_t)) << fail_msg.TraceFailMsgs();

    ASSERT_EQ(rsvd_src[1].shape().size(), 3UL);
    EXPECT_EQ(rsvd_src[1].shape()[0], 2UL);
    EXPECT_EQ(rsvd_src[1].shape()[1], 3UL);
    EXPECT_EQ(rsvd_src[1].shape()[2], keepdim);
    ASSERT_EQ(rsvd_src[2].shape().size(), 2UL);
    EXPECT_EQ(rsvd_src[2].shape()[0], keepdim);
    EXPECT_EQ(rsvd_src[2].shape()[1], 5UL);

    ASSERT_EQ(rsvd_src_t[1].shape().size(), 2UL);
    EXPECT_EQ(rsvd_src_t[1].shape()[0], 5UL);
    EXPECT_EQ(rsvd_src_t[1].shape()[1], keepdim);
    ASSERT_EQ(rsvd_src_t[2].shape().size(), 3UL);
    EXPECT_EQ(rsvd_src_t[2].shape()[0], keepdim);
    EXPECT_EQ(rsvd_src_t[2].shape()[1], 2UL);
    EXPECT_EQ(rsvd_src_t[2].shape()[2], 3UL);
  }

  bool CheckResult(const std::string& case_name, const cytnx_uint64& keepdim,
                   const cytnx_uint64& power_iteration) {
    std::string src_file_name = src_data_root + case_name + ".cytnx";
    std::string ans_file_name = ans_data_root + case_name + ".cytnx";
    std::string rec_file_name = ans_data_root + case_name + "_reconstructed.cytnx";
    UniTensor src_T = UniTensor::Load(src_file_name).to(Device.cuda);
    UniTensor ans_T = UniTensor::Load(ans_file_name).to(Device.cuda);
    UniTensor rec_T = UniTensor::Load(rec_file_name).to(Device.cuda);

    std::vector<UniTensor> rsvds =
      linalg::Rsvd(src_T, keepdim, 0, true, true, 0, 0, 2, 1, power_iteration, 0);

    if (!CheckLabels(src_T, rsvds)) {
      fail_msg.AppendMsg("The output labels are wrong. ", __func__, __LINE__);
      return false;
    }
    if (!SingularValsCorrect(rsvds[0], ans_T)) {
      fail_msg.AppendMsg("The singular values are wrong. ", __func__, __LINE__);
      return false;
    }
    if (!ReComposeCheck(rec_T, rsvds)) {
      fail_msg.AppendMsg("The result is wrong after recomposing, T is not equal to USV*.", __func__,
                         __LINE__);
      return false;
    }

    return true;
  }

  bool CheckResult(const std::string& case_name, const cytnx_uint64& keepdim,
                   const std::vector<cytnx_uint64> min_blockdim,
                   const cytnx_uint64& power_iteration) {
    std::string src_file_name = src_data_root + case_name + ".cytnx";
    std::string ans_file_name = ans_data_root + case_name + "_minblockdim.cytnx";
    std::string rec_file_name = ans_data_root + case_name + "_minblockdim_reconstructed.cytnx";
    UniTensor src_T = UniTensor::Load(src_file_name).to(Device.cuda);
    UniTensor ans_T = UniTensor::Load(ans_file_name).to(Device.cuda);
    UniTensor rec_T = UniTensor::Load(rec_file_name).to(Device.cuda);

    std::vector<UniTensor> rsvds =
      linalg::Rsvd(src_T, keepdim, min_blockdim, 0., true, true, 0, 0, 2, 1, power_iteration, 0);

    if (!CheckLabels(src_T, rsvds)) {
      fail_msg.AppendMsg("The output labels are wrong. ", __func__, __LINE__);
      return false;
    }
    if (!SingularValsCorrect(rsvds[0], ans_T)) {
      fail_msg.AppendMsg("The singular values are wrong. ", __func__, __LINE__);
      return false;
    }
    if (!ReComposeCheck(rec_T, rsvds)) {
      fail_msg.AppendMsg("The result is wrong after recomposing, T is not equal to USV*.", __func__,
                         __LINE__);
      return false;
    }

    return true;
  }

  UniTensor BuildCombinedBlockFermionicTensorWithSignflip() {
    std::vector<std::vector<cytnx_int64>> qnums = {{0, 0}, {1, 1}};
    auto syms = std::vector<Symmetry>{Symmetry::FermionParity(), Symmetry(SymmetryType::U)};
    Bond l1 = Bond(BD_IN, qnums, {2, 3}, syms);
    Bond l2 = Bond(BD_IN, qnums, {1, 2}, syms);
    Bond r1 = l1.redirect();
    Bond r2 = l2.redirect();

    UniTensor T({l1, l2, r1, r2}, {"l1", "l2", "r1", "r2"}, 2, Type.Double, Device.cuda, false);
    random::uniform_(T, -1, 1, 0);
    std::vector<std::vector<cytnx_int64>> perms = {
      {1, 0, 2, 3},
      {0, 2, 1, 3},
      {2, 1, 0, 3},
      {0, 1, 3, 2},
    };

    for (const auto& p : perms) {
      UniTensor cand = T.permute(p);
      bool has_signflip = false;
      for (auto sf : cand.signflip()) {
        if (sf) {
          has_signflip = true;
          break;
        }
      }
      if (has_signflip) return cand;
    }
    return T.permute({1, 0, 2, 3});
  }
}  // namespace RsvdTest
