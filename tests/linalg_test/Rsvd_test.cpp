#include <gtest/gtest.h>

#include "cytnx.hpp"
#include "test_tools.h"

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
  TEST(Rsvd, dense_one_elem) {
    std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
    fail_msg.Init(test_case_name);
    int size = 1;
    std::vector<Bond> bonds = {Bond(size), Bond(size), Bond(size)};
    int rowrank = 1;
    bool is_diag = false;
    auto labels = std::vector<std::string>();
    auto T = UniTensor(bonds, labels, rowrank, cytnx::Type.Double, cytnx::Device.cpu, is_diag);
    random::uniform_(T, -10, 0, 0);
    std::vector<UniTensor> rsvds = linalg::Rsvd(T, 1);
    EXPECT_TRUE(CheckLabels(T, rsvds)) << fail_msg.TraceFailMsgs();
    EXPECT_TRUE(ReComposeCheck(T, rsvds)) << fail_msg.TraceFailMsgs();
    EXPECT_DOUBLE_EQ(rsvds[0].at<double>({0}), std::abs(T.at<double>({0, 0, 0})))
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
  // TEST(Rsvd, dense_nondiag_test) {
  //   std::vector<std::string> case_list = {"dense_nondiag_C128", "dense_nondiag_F64"};
  //   for (const auto& case_name : case_list) {
  //     std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
  //     fail_msg.Init(test_case_name + ", " + case_name);
  //     EXPECT_TRUE(CheckResult(case_name, 4, 2)) << fail_msg.TraceFailMsgs();
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
  TEST(Rsvd, err_dense_diag_test) {
    int size = 5;
    std::vector<Bond> bonds = {Bond(size), Bond(size)};
    int rowrank = 1;
    bool is_diag = true;
    auto labels = std::vector<std::string>();
    auto T = UniTensor(bonds, labels, rowrank, cytnx::Type.Double, cytnx::Device.cpu, is_diag);
    random::uniform_(T, 0, 10, 0);
    EXPECT_THROW({ std::vector<UniTensor> rsvds = linalg::Rsvd(T, 2); }, std::logic_error);
  }

  /*=====test info=====
  describe:Test Dense UniTensor with exponentially decaying singular values.
  input:
    T:Dense UniTensor with real or complex real type.
    is_U:true
    is_VT:true
  ====================*/
  TEST(Rsvd, dense_exp_svals_test) {
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
  TEST(Rsvd, dense_exp_svals_no_power_iteration_test) {
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
  TEST(Rsvd, U1_exp_svals_test) {
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
  TEST(Rsvd, U1_exp_svals_minblockdim_test) {
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
  TEST(Rsvd, block_fermionic_test) {
    std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
    fail_msg.Init(test_case_name);
    UniTensor src_T = BuildCombinedBlockFermionicTensorWithSignflip();
    std::vector<UniTensor> rsvds = linalg::Rsvd(src_T, 10, 0., true, true, 0, 0, 2, 1, 2, 0);
    EXPECT_TRUE(CheckLabels(src_T, rsvds)) << fail_msg.TraceFailMsgs();
    // Note: For truncated SVD, exact recomposition is not expected due to truncation
    // EXPECT_TRUE(ReComposeCheck(src_T, rsvds)) << fail_msg.TraceFailMsgs();
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
    UD.set_labels({"0", "1", "9"});
    UD.permute_({2, 0, 1}, 1);
    auto UUD = Contract(U, UD);
  }

  bool CheckResult(const std::string& case_name, const cytnx_uint64& keepdim,
                   const cytnx_uint64& power_iteration) {
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

    // Do Rsvd
    std::vector<UniTensor> rsvds =
      linalg::Rsvd(src_T, keepdim, 0, true, true, 0, 0, 2, 1, power_iteration, 0);

    // check labels
    if (!CheckLabels(src_T, rsvds)) {
      fail_msg.AppendMsg("The output labels are wrong. ", __func__, __LINE__);
      return false;
    }

    // check answer
    if (!SingularValsCorrect(rsvds[0], ans_T)) {
      fail_msg.AppendMsg("The singular values are wrong. ", __func__, __LINE__);
      return false;
    }

    // check recompose [M - USV*]
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

    // Do Rsvd
    std::vector<UniTensor> rsvds =
      linalg::Rsvd(src_T, keepdim, min_blockdim, 0., true, true, 0, 0, 2, 1, power_iteration, 0);

    // check labels
    if (!CheckLabels(src_T, rsvds)) {
      fail_msg.AppendMsg("The output labels are wrong. ", __func__, __LINE__);
      return false;
    }

    // check answer
    if (!SingularValsCorrect(rsvds[0], ans_T)) {
      fail_msg.AppendMsg("The singular values are wrong. ", __func__, __LINE__);
      return false;
    }

    // check recompose [M - USV*]
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

    UniTensor T({l1, l2, r1, r2}, {"l1", "l2", "r1", "r2"}, 2, Type.Double, Device.cpu, false);
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
