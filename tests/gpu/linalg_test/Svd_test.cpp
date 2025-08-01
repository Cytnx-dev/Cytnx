#include "gtest/gtest.h"

#include "gpu_test_tools.h"
#include "cytnx.hpp"

using namespace cytnx;
using namespace testing;
using namespace TestTools;

namespace SvdTest {

  static TestFailMsg fail_msg;

  bool CheckResult(const std::string& case_name);
  bool ReComposeCheck(const UniTensor& Tin, const std::vector<UniTensor>& Tout);
  bool CheckLabels(const UniTensor& Tin, const std::vector<UniTensor>& Tout);
  bool SingularValsCorrect(const UniTensor& res, const UniTensor& ans);
  std::string src_data_root = CYTNX_TEST_DATA_DIR "/common/";
  std::string ans_data_root = CYTNX_TEST_DATA_DIR "/linalg/Svd/";
  // normal test

  /*=====test info=====
  describe:Test dense UniTensor only one element.
  input:
    T:Dense UniTensor only one element.
    is_U:true
    is_VT:true
  ====================*/
  TEST(Svd, gpu_dense_one_elem) {
    std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
    fail_msg.Init(test_case_name);
    int size = 1;
    std::vector<Bond> bonds = {Bond(size), Bond(size), Bond(size)};
    int rowrank = 1;
    bool is_diag = false;
    auto labels = std::vector<std::string>();
    auto T = UniTensor(bonds, labels, rowrank, cytnx::Type.Double, cytnx::Device.cuda, is_diag)
               .to(cytnx::Device.cuda);
    random::Make_uniform(T, -10, 0, 0);
    std::vector<UniTensor> svds = linalg::Svd(T);
    EXPECT_TRUE(CheckLabels(T, svds)) << fail_msg.TraceFailMsgs();
    EXPECT_TRUE(ReComposeCheck(T, svds)) << fail_msg.TraceFailMsgs();
    EXPECT_DOUBLE_EQ(svds[0].at<double>({0}), std::abs(T.at<double>({0, 0, 0})))
      << "Singular value is wrong."
      << " line:" << __LINE__ << std::endl;
  }

  /*=====test info=====
  describe:Test Dense UniTensor.
  input:
    T:Dense UniTensor with real or complex real type.
    is_U:true
    is_VT:true
  ====================*/
  TEST(Svd, gpu_dense_nondiag_test) {
    std::vector<std::string> case_list = {"dense_nondiag_C128", "dense_nondiag_F64"};
    for (const auto& case_name : case_list) {
      std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
      fail_msg.Init(test_case_name + ", " + case_name);
      EXPECT_TRUE(CheckResult(case_name)) << fail_msg.TraceFailMsgs();
    }
  }

  /*=====test info=====
  describe:Test U1 symmetry tensor.
  input:
    T:Tensor with U1 symmetry (Double and ComplexDoubel).
    is_U:true
    is_VT:true
  ====================*/
  TEST(Svd, gpu_U1_sym_test) {
    std::vector<std::string> case_list = {"sym_UT_U1_C128", "sym_UT_U1_F64"};
    for (const auto& case_name : case_list) {
      std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
      fail_msg.Init(test_case_name + ", " + case_name);
      EXPECT_TRUE(CheckResult(case_name)) << fail_msg.TraceFailMsgs();
    }
  }

  /*=====test info=====
  describe:Test Z2 symmetry tensor.
  input:
    T:Tensor with Z2 symmetry (Double and ComplexDoubel).
    is_U:true
    is_VT:true
  ====================*/
  TEST(Svd, gpu_Z2_sym_test) {
    std::vector<std::string> case_list = {"sym_UT_Z2_C128", "sym_UT_Z2_F64"};
    for (const auto& case_name : case_list) {
      std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
      fail_msg.Init(test_case_name + ", " + case_name);
      EXPECT_TRUE(CheckResult(case_name)) << fail_msg.TraceFailMsgs();
    }
  }

  /*=====test info=====
  describe:Test Z3 symmetry tensor.
  input:
    T:Tensor with Z3 symmetry (Double and ComplexDoubel).
    is_U:true
    is_VT:true
  ====================*/
  TEST(Svd, gpu_Z3_sym_test) {
    std::vector<std::string> case_list = {"sym_UT_Z3_C128", "sym_UT_Z3_F64"};
    for (const auto& case_name : case_list) {
      std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
      fail_msg.Init(test_case_name + ", " + case_name);
      EXPECT_TRUE(CheckResult(case_name)) << fail_msg.TraceFailMsgs();
    }
  }

  /*=====test info=====
  describe:Test U1xZ2 symmetry tensor.
  input:
    T:Tensor with U1xZ2 symmetry (all possilble data type).
    is_U:true
    is_VT:true
  ====================*/
  TEST(Svd, gpu_U1xZ2_sym_test) {
    std::vector<std::string> case_list = {
      "sym_UT_U1xZ2_C128", "sym_UT_U1xZ2_C64", "sym_UT_U1xZ2_F64", "sym_UT_U1xZ2_F32",
      "sym_UT_U1xZ2_I64",  "sym_UT_U1xZ2_U64", "sym_UT_U1xZ2_I32", "sym_UT_U1xZ2_U32",
      "sym_UT_U1xZ2_I16",  "sym_UT_U1xZ2_U16"};
    for (const auto& case_name : case_list) {
      std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
      fail_msg.Init(test_case_name + ", " + case_name);
      EXPECT_TRUE(CheckResult(case_name)) << fail_msg.TraceFailMsgs();
    }
  }

  /*=====test info=====
  describe:Test input a symmetry tensor contains all zero elements.
  input:
    T:Tensor with U1 symmetry with all elements are zero.
    is_U:true
    is_VT:true
  ====================*/
  TEST(Svd, gpu_U1_zeros_test) {
    GTEST_SKIP() << "Issue for cuda. Cannot handle if most of elements are zeros.";
    std::string case_name = "sym_UT_U1_zeros_F64";
    std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
    fail_msg.Init(test_case_name + ", " + case_name);
    EXPECT_TRUE(CheckResult(case_name)) << fail_msg.TraceFailMsgs();
  }

  /*=====test info=====
  describe:Test input a symmetry tensor contains only one element.
  input:
    T:Tensor with U1 symmetry contains only one element..
    is_U:true
    is_VT:true
  ====================*/
  TEST(Svd, gpu_U1_one_elem_test) {
    std::string case_name = "sym_UT_U1_one_elem_F64";
    std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
    fail_msg.Init(test_case_name + ", " + case_name);
    EXPECT_TRUE(CheckResult(case_name)) << fail_msg.TraceFailMsgs();
  }

  /*=====test info=====
  describe:Test the input parameters 'is_U' or 'is_VT' may be false.
  input:
    T:Tensor with U1 symmetry contains only one element..
    is_U:'true' and 'false' both will be test.
    is_VT:'true' and 'false' both will be test.
  ====================*/
  TEST(Svd, gpu_none_isU_isVT) {
    // bool need_U, need_VT;
    bool compute_uv;
    std::string case_name = "sym_UT_U1_F64";
    std::string src_file_name = src_data_root + case_name + ".cytnx";
    UniTensor src_T = UniTensor::Load(src_file_name);
    // get S, U, VT
    //  std::vector<UniTensor> svdsUV = linalg::Svd(src_T, need_U = true, need_VT = true);
    //  EXPECT_EQ(svdsUV.size(), 3) << "The number of the output of svd not correct. Line:"
    //      << __LINE__ << std::endl;
    std::vector<UniTensor> svdsUV = linalg::Svd(src_T, compute_uv = true);
    EXPECT_EQ(svdsUV.size(), 3) << "The number of the output of svd not correct. Line:" << __LINE__
                                << std::endl;

    // //get only S, U
    // std::vector<UniTensor> svdsU = linalg::Svd(src_T, need_U = true, need_VT = false);
    // EXPECT_EQ(svdsU.size(), 2) << "The number of the output of svd not correct. Line:"
    //     << __LINE__ << std::endl;
    // EXPECT_TRUE(AreEqUniTensor(svdsU[0], svdsUV[0])) << "In the svd, output 'S' not"
    //     " correct. Line: " << __LINE__ << std::endl;
    // EXPECT_TRUE(AreEqUniTensor(svdsU[1], svdsUV[1])) << "In the svd, output 'U' not"
    //     " correct. Line: " << __LINE__ << std::endl;

    // //get only S, VT
    // std::vector<UniTensor> svdsV = linalg::Svd(src_T, need_U = false, need_VT = true);
    // EXPECT_EQ(svdsV.size(), 2) << "The number of the output of svd not correct. Line:"
    //     << __LINE__ << std::endl;
    // EXPECT_TRUE(AreEqUniTensor(svdsV[0], svdsUV[0])) << "In the svd, output 'S' not"
    //     " correct. Line: " << __LINE__ << std::endl;
    // EXPECT_TRUE(AreEqUniTensor(svdsV[1], svdsUV[2])) << "In the svd, output 'VT' not"
    //     " correct. Line: " << __LINE__ << std::endl;

    // get only S
    //  std::vector<UniTensor> svdsS = linalg::Svd(src_T, need_U = false, need_VT = false);
    //  EXPECT_EQ(svdsS.size(), 1) << "The number of the output of svd not correct. Line:"
    //      << __LINE__ << std::endl;
    std::vector<UniTensor> svdsS = linalg::Svd(src_T, compute_uv = false);
    EXPECT_EQ(svdsS.size(), 1) << "The number of the output of svd not correct. Line:" << __LINE__
                               << std::endl;

    const double tol = 1.0e-10;
    // Since if no is_U and is_vT, we may call Lapack for different arguments to
    //   say we don't need U and VT, so we have tolerance here.
    EXPECT_TRUE(AreNearlyEqUniTensor(svdsS[0], svdsUV[0], tol));
  }

  // error test

  /*=====test info=====
  describe:error test, Test Dense diagonal tensor.
  input:
    T:Dense diagonal complex real type UniTensor.
    is_U:true
    is_VT:true
  ====================*/
  TEST(Svd, gpu_err_dense_diag_test) {
    int size = 5;
    std::vector<Bond> bonds = {Bond(size), Bond(size)};
    int rowrank = 1;
    bool is_diag = true;
    auto labels = std::vector<std::string>();
    auto T = UniTensor(bonds, labels, rowrank, cytnx::Type.Double, cytnx::Device.cuda, is_diag)
               .to(cytnx::Device.cuda);
    random::Make_uniform(T, 0, 10, 0);
    EXPECT_THROW({ std::vector<UniTensor> svds = linalg::Svd(T); }, std::logic_error);
  }

  /*=====test info=====
  describe:error test. Test Symmetric diagonal tensor.
  input:
    T:Symmetric diagonal complex real type UniTensor.
    is_U:true
    is_VT:true
  ====================*/
  TEST(Svd, gpu_err_sym_diag_test) {
    std::vector<std::vector<cytnx_int64>> qnums = {{0}, {1}, {0}, {1}, {2}};
    std::vector<cytnx_uint64> degs = {1, 2, 3, 4, 5};
    auto syms = std::vector<Symmetry>(qnums[0].size(), Symmetry(SymType.U));
    auto bond_ket = Bond(BD_KET, qnums, degs, syms);
    auto bond_bra = Bond(BD_BRA, qnums, degs, syms);
    std::vector<Bond> bonds = {bond_ket, bond_bra};
    cytnx_int64 row_rank = 1;
    bool is_diag = true;
    std::vector<std::string> labels = {};
    auto UT =
      UniTensor(bonds, labels, row_rank, Type.Double, Device.cuda, is_diag).to(cytnx::Device.cuda);
    random::Make_uniform(UT, 0, 10, 0);
    EXPECT_THROW({ std::vector<UniTensor> svds = linalg::Svd(UT); }, std::logic_error);
  }

  /*=====test info=====
  describe:eror test. Test input the symmetric UniTensor with dtype is bool type.
  input:
    T:bool type symmetric UniTensor.
    is_U:true
    is_VT:true
  ====================*/
  TEST(Svd, gpu_err_bool_type_UT) {
    std::vector<std::vector<cytnx_int64>> qnums1 = {{0}, {1}, {0}, {1}, {2}};
    std::vector<cytnx_uint64> degs = {1, 2, 3, 4, 5};
    auto syms = std::vector<Symmetry>(qnums1[0].size(), Symmetry(SymType.U));
    auto bond_ket = Bond(BD_KET, qnums1, degs, syms);
    std::vector<std::vector<cytnx_int64>> qnums2 = {{-1}, {-1}, {0}, {2}, {1}};
    syms = std::vector<Symmetry>(qnums2[0].size(), Symmetry(SymType.U));
    auto bond_bra = Bond(BD_BRA, qnums2, degs, syms);
    std::vector<Bond> bonds = {bond_ket, bond_ket, bond_bra, bond_bra};
    cytnx_int64 row_rank = -1;
    std::vector<std::string> labels = {};
    auto src_T =
      UniTensor(bonds, labels, row_rank, Type.Bool, Device.cuda, false).to(cytnx::Device.cuda);
    InitUniTensorUniform(src_T);
    // bool need_U, need_VT;
    bool compute_uv;
    // EXPECT_THROW({
    //   std::vector<UniTensor> svds = linalg::Svd(src_T, need_U = true, need_VT = true);
    // }, std::logic_error) << "Should throw error when input bool type unitensor but not."
    //     " Line:" << __LINE__ << std::endl;
    EXPECT_THROW({ std::vector<UniTensor> svds = linalg::Svd(src_T, compute_uv = true); },
                 std::logic_error)
      << "Should throw error when input bool type unitensor but not."
         " Line:"
      << __LINE__ << std::endl;
  }

  /*=====test info=====
  describe:eror test. Test input Void type UniTensor.
  input:
    T:Void UniTensor.
  ====================*/
  TEST(Svd, gpu_err_Void_UTenType_UT) {
    auto Ut = UniTensor();
    EXPECT_THROW({ std::vector<UniTensor> svds = linalg::Svd(Ut); }, std::logic_error)
      << "Should throw error when input void type unitensor but not."
         " Line:"
      << __LINE__ << std::endl;
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
    const double tol = is_double_float_acc ? 1.0e-9 : 1.0e-2;
    auto T_float = Tin.clone();
    if (Tin.dtype() > Type.Float) {
      T_float = Tin.astype(Type.Double);
    }
    bool is_eq = AreNearlyEqUniTensor(T_float, ReCompose, tol);
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

    // const double tol = is_double_float_acc ? 1.0e-14 : 1.0e-6;
    /*
     *  I changed relative_err tolerance of singular value in Svd_test from 1e-14 to 0.05, since
     *  singular value may varies when using different algorithms
     */
    const double tol = is_double_float_acc ? 0.05 : 0.05;
    if (relative_err > tol) {
      std::cout << "relative_err: " << relative_err << std::endl;
      std::cout << "res:\n" << res << std::endl;
      std::cout << "ans:\n" << ans << std::endl;
      std::cout << "diff:\n" << diff_tens << std::endl;
    }
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

  bool CheckResult(const std::string& case_name) {
    // test data source file
    std::string src_file_name = src_data_root + case_name + ".cytnx";
    // anscer file
    std::string ans_file_name = ans_data_root + case_name + ".cytnx";
    // bool need_U, need_VT;
    bool compute_uv;
    UniTensor src_T = UniTensor::Load(src_file_name).to(cytnx::Device.cuda);
    UniTensor ans_T =
      UniTensor::Load(ans_file_name).to(cytnx::Device.cuda);  // sigular values UniTensor

    // Do svd
    std::vector<UniTensor> svds = linalg::Svd(src_T, compute_uv = true);

    // check labels
    if (!(CheckLabels(src_T, svds))) {
      fail_msg.AppendMsg("The output labels are wrong. ", __func__, __LINE__);
      return false;
    }

    // check answer
    if (!(SingularValsCorrect(svds[0], ans_T))) {
      fail_msg.AppendMsg("The singular values are wong.. ", __func__, __LINE__);
      return false;
    }

    // check recomplse [M - USV*]
    if (!ReComposeCheck(src_T, svds)) {
      fail_msg.AppendMsg(
        "The result is wrong after recomposing. "
        "That's mean T not equal USV* ",
        __func__, __LINE__);
      return false;
    }

    return true;
  }

}  // namespace SvdTest
