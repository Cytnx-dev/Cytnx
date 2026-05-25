#include "cytnx.hpp"
#include <gtest/gtest.h>
#include "../test_tools.h"

using namespace cytnx;
using namespace testing;
using namespace TestTools;

namespace GesvdTruncateTest {

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
  TEST(Gesvd_truncate, dense_one_elem) {
    std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
    fail_msg.Init(test_case_name);
    int size = 1;
    std::vector<Bond> bonds = {Bond(size), Bond(size), Bond(size)};
    int rowrank = 1;
    bool is_diag = false;
    auto labels = std::vector<std::string>();
    auto T = UniTensor(bonds, labels, rowrank, cytnx::Type.Double, cytnx::Device.cpu, is_diag);
    random::uniform_(T, -10, 0, 0);
    std::vector<UniTensor> Svds = linalg::Gesvd_truncate(T, 1);
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
  // TEST(Gesvd_truncate, dense_nondiag_test) {
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
  TEST(Gesvd_truncate, err_dense_diag_test) {
    int size = 5;
    std::vector<Bond> bonds = {Bond(size), Bond(size)};
    int rowrank = 1;
    bool is_diag = true;
    auto labels = std::vector<std::string>();
    auto T = UniTensor(bonds, labels, rowrank, cytnx::Type.Double, cytnx::Device.cpu, is_diag);
    random::uniform_(T, 0, 10, 0);
    EXPECT_THROW({ std::vector<UniTensor> Svds = linalg::Gesvd_truncate(T, 2); }, std::logic_error);
  }

  /*=====test info=====
  describe:Test Dense UniTensor with exponentially decaying singular values.
  input:
    T:Dense UniTensor with real or complex real type.
    is_U:true
    is_VT:true
  ====================*/
  TEST(Gesvd_truncate, dense_exp_svals_test) {
    std::vector<std::string> case_list = {"dense_nondiag_exp_Svals_C128",
                                          "dense_nondiag_exp_Svals_F64"};
    for (const auto& case_name : case_list) {
      std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
      fail_msg.Init(test_case_name + ", " + case_name);
      EXPECT_TRUE(CheckResult(case_name, 5)) << fail_msg.TraceFailMsgs();
    }
  }

  /*=====test info=====
  describe:Test U(1) UniTensor with exponentially decaying singular values.
  input:
    T:U(1) UniTensor with real or complex real type.
    is_U:true
    is_VT:true
  ====================*/
  TEST(Gesvd_truncate, U1_exp_svals_test) {
    std::vector<std::string> case_list = {"sym_UT_U1_exp_Svals_C128", "sym_UT_U1_exp_Svals_F64"};
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
  TEST(Gesvd_truncate, U1_exp_svals_minblockdim_test) {
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
    // std::cout << relative_err << std::endl;

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

    // Do Gesvd_truncate
    std::vector<UniTensor> Svds = linalg::Gesvd_truncate(src_T, keepdim, 0., true, true, 0);

    // check labels
    if (!(CheckLabels(src_T, Svds))) {
      fail_msg.AppendMsg("The output labels are wrong. ", __func__, __LINE__);
      return false;
    }

    // check answer
    if (!(SingularValsCorrect(Svds[0], ans_T))) {
      fail_msg.AppendMsg("The singular values are wrong.. ", __func__, __LINE__);
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

    // Do Gesvd_truncate
    std::vector<UniTensor> Svds =
      linalg::Gesvd_truncate(src_T, keepdim, min_blockdim, 0., true, true, 0);

    // check labels
    if (!(CheckLabels(src_T, Svds))) {
      fail_msg.AppendMsg("The output labels are wrong. ", __func__, __LINE__);
      return false;
    }

    ans_T.print_diagram();
    Svds[0].print_diagram();
    // check answer
    if (!(SingularValsCorrect(Svds[0], ans_T))) {
      fail_msg.AppendMsg("The singular values are wrong.. ", __func__, __LINE__);
      return false;
    }
    Svds[1].print_diagram();
    Svds[2].print_diagram();

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
  describe:Tensor-level Gesvd_truncate must work for all (is_U, is_vT) combinations.
  ====================*/
  // Singular values are gauge-invariant; compare them with a relative tolerance (LAPACK gesvd
  // can differ slightly across jobu/jobvt modes). U/vT are sign/gauge dependent across modes,
  // so we only assert their truncated shapes.
  static bool SingularValsClose(const Tensor& a, const Tensor& b, double rtol = 1e-14) {
    Tensor ad = a.astype(Type.Double).contiguous();
    Tensor bd = b.astype(Type.Double).contiguous();
    if (ad.shape() != bd.shape()) return false;
    for (cytnx_uint64 i = 0; i < ad.storage().size(); ++i) {
      double x = ad.storage().at<double>(i), y = bd.storage().at<double>(i);
      if (std::abs(x - y) > rtol * (1.0 + std::abs(y))) return false;
    }
    return true;
  }

  TEST(Gesvd_truncate, flag_combinations_dense) {
    for (auto dtype : {Type.Double, Type.ComplexDouble}) {
      Tensor T = Tensor({6, 5}, dtype);
      InitTensorUniform(T, /*seed=*/2);
      const cytnx_uint64 keep = 3;

      // reference singular values (both matrices requested)
      std::vector<Tensor> ref = linalg::Gesvd_truncate(T, keep, 0., true, true, 0, 1);
      ASSERT_EQ(ref.size(), 3u);
      const cytnx_uint64 k = ref[0].shape()[0];
      EXPECT_EQ(k, keep);
      EXPECT_EQ(ref[1].shape(), std::vector<cytnx_uint64>({6, k}));
      EXPECT_EQ(ref[2].shape(), std::vector<cytnx_uint64>({k, 5}));

      for (bool is_U : {false, true}) {
        for (bool is_vT : {false, true}) {
          std::vector<Tensor> out = linalg::Gesvd_truncate(T, keep, 0., is_U, is_vT, 0, 1);
          const cytnx_uint64 expected = 1u + (is_U ? 1u : 0u) + (is_vT ? 1u : 0u);
          ASSERT_EQ(out.size(), expected) << "is_U=" << is_U << " is_vT=" << is_vT;

          EXPECT_TRUE(SingularValsClose(out[0], ref[0]))
            << "S mismatch, is_U=" << is_U << " is_vT=" << is_vT;

          cytnx_uint64 idx = 1;
          if (is_U) {
            EXPECT_EQ(out[idx].shape(), std::vector<cytnx_uint64>({6, k}));
            ++idx;
          }
          if (is_vT) {
            // the truncated vT must have k rows
            EXPECT_EQ(out[idx].shape(), std::vector<cytnx_uint64>({k, 5}));
          }
        }
      }
    }
  }
}  // namespace GesvdTruncateTest
