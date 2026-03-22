#include "cytnx.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <map>
#include "../test_tools.h"

using namespace cytnx;
using namespace testing;
using namespace TestTools;

namespace RsvdNoTruncateTest {

  static TestFailMsg fail_msg;

  bool CheckResult(const std::string& case_name, const cytnx_uint64& keepdim,
                   const cytnx_uint64& power_iteration);
  bool ReComposeCheck(const UniTensor& Tin, const std::vector<UniTensor>& Tout);
  bool CheckLabels(const UniTensor& Tin, const std::vector<UniTensor>& Tout);
  bool SingularValsCorrect(const UniTensor& res, const UniTensor& ans);
  bool CheckAgainstGesvd(const UniTensor& src_T, const cytnx_uint64& keepdim,
                         const cytnx_uint64& power_iteration, bool is_U = true, bool is_vT = true,
                         cytnx_uint64 mindim = 1, cytnx_uint64 oversampling_summand = 0,
                         double oversampling_factor = 0.0);
  UniTensor BuildCombinedBlockTensor();
  UniTensor BuildCombinedBlockTensorU1xZ2();
  UniTensor BuildCombinedBlockFermionicTensor();
  UniTensor BuildCombinedBlockFermionicTensorWithSignflip();
  std::vector<cytnx_uint64> ExpectedKeptDims(const UniTensor& Tin, const UniTensor& full_svals,
                                             cytnx_uint64 keepdim);
  bool CheckPerBlockLeadingSvals(const UniTensor& rsvd_svals, const UniTensor& gesvd_svals,
                                 double tol);
  cytnx_uint64 FindKeepdimForCategory(const UniTensor& Tin, const UniTensor& full_svals,
                                      bool require_full, bool require_one_kept);
  void CheckCategoryCoverage(const std::vector<cytnx_uint64>& full_dims,
                             const std::vector<cytnx_uint64>& kept_dims, bool& has_full,
                             bool& has_trunc, bool& has_one_kept);
  std::string src_data_root = CYTNX_TEST_DATA_DIR "/common/";
  std::string ans_data_root = CYTNX_TEST_DATA_DIR "/linalg/Rsvd/";
  // normal test

  /*=====test info=====
  describe:Test dense UniTensor only one element.
  input:
    T:Dense UniTensor only one element.
    is_U:true
    is_VT:true
  ====================*/
  TEST(Rsvd_notruncate, dense_one_elem) {
    std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
    fail_msg.Init(test_case_name);
    int size = 1;
    std::vector<Bond> bonds = {Bond(size), Bond(size), Bond(size)};
    int rowrank = 1;
    bool is_diag = false;
    auto labels = std::vector<std::string>();
    auto T = UniTensor(bonds, labels, rowrank, cytnx::Type.Double, cytnx::Device.cpu, is_diag);
    random::uniform_(T, -10, 0, 0);
    std::vector<UniTensor> Rsvds = linalg::Rsvd_notruncate(T, 1);
    EXPECT_TRUE(CheckLabels(T, Rsvds)) << fail_msg.TraceFailMsgs();
    EXPECT_TRUE(ReComposeCheck(T, Rsvds)) << fail_msg.TraceFailMsgs();
    EXPECT_EQ(Rsvds[0].at<double>({0}), std::abs(T.at<double>({0, 0, 0})))
      << "Singular value is wrong."
      << " line:" << __LINE__ << std::endl;
  }

  /*=====test info=====
  describe:error test, Test Dense diagonal tensor.
  input:
    T:Dense diagonal complex real type UniTensor.
    is_U:true
    is_VT:true
  ====================*/
  TEST(Rsvd_notruncate, err_dense_diag_test) {
    int size = 5;
    std::vector<Bond> bonds = {Bond(size), Bond(size)};
    int rowrank = 1;
    bool is_diag = true;
    auto labels = std::vector<std::string>();
    auto T = UniTensor(bonds, labels, rowrank, cytnx::Type.Double, cytnx::Device.cpu, is_diag);
    random::uniform_(T, 0, 10, 0);
    EXPECT_THROW({ std::vector<UniTensor> Rsvds = linalg::Rsvd_notruncate(T, 2); },
                 std::logic_error);
  }

  /*=====test info=====
  describe:error test, Test symmetric diagonal UniTensor.
  input:
    T:Symmetric diagonal UniTensor.
  ====================*/
  TEST(Rsvd_notruncate, err_sym_diag_test) {
    Bond bond_ket = Bond(BD_KET, {Qs(0), Qs(1), Qs(2)}, {2, 1, 2});
    Bond bond_bra = bond_ket.redirect();
    UniTensor UT = UniTensor({bond_ket, bond_bra}, {}, 1, Type.Double, Device.cpu, true);
    EXPECT_THROW({ std::vector<UniTensor> Rsvds = linalg::Rsvd_notruncate(UT, 2); },
                 std::logic_error);
  }

  /*=====test info=====
  describe:error test, Test rank-1 UniTensor.
  input:
    T:rank-1 dense UniTensor.
  ====================*/
  TEST(Rsvd_notruncate, err_rank1_unitensor) {
    UniTensor T = UniTensor({Bond(8)}, {"x"}, 0, Type.Double, Device.cpu, false);
    random::uniform_(T, 0, 10, 0);
    EXPECT_THROW({ std::vector<UniTensor> Rsvds = linalg::Rsvd_notruncate(T, 2); },
                 std::logic_error);
  }

  /*=====test info=====
  describe:Test Dense UniTensor with exponentially decaying singular values.
  input:
    T:Dense UniTensor with real or complex real type.
    is_U:true
    is_VT:true
  ====================*/
  TEST(Rsvd_notruncate, dense_exp_svals_test) {
    std::vector<std::string> case_list = {"dense_nondiag_exp_Svals_C128",
                                          "dense_nondiag_exp_Svals_F64"};
    for (const auto& case_name : case_list) {
      std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
      fail_msg.Init(test_case_name + ", " + case_name);
      EXPECT_TRUE(CheckResult(case_name, 15, 2)) << fail_msg.TraceFailMsgs();
    }
  }

  /*=====test info=====
  describe:Test Dense UniTensor with exponentially decaying singular values. No power iteration in
  Rsvd_notruncate. input: T:Dense UniTensor with real or complex real type. is_U:true is_VT:true
  ====================*/
  TEST(Rsvd_notruncate, dense_exp_svals_no_power_iteration_test) {
    std::vector<std::string> case_list = {"dense_nondiag_exp_Svals_C128",
                                          "dense_nondiag_exp_Svals_F64"};
    for (const auto& case_name : case_list) {
      std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
      fail_msg.Init(test_case_name + ", " + case_name);
      EXPECT_TRUE(CheckResult(case_name, 15, 0)) << fail_msg.TraceFailMsgs();
    }
  }

  /*=====test info=====
  describe:Test tagged Dense UniTensor output bond types in Rsvd_notruncate.
  input:
    T:Tagged dense UniTensor with rowrank=2.
    is_U:true
    is_VT:true
  ====================*/
  TEST(Rsvd_notruncate, dense_tagged_output_bond_types) {
    std::vector<Bond> bonds = {Bond(2), Bond(3), Bond(4), Bond(5)};
    std::vector<std::string> labels = {"a", "b", "c", "d"};
    UniTensor T(bonds, labels, 2, Type.Double, Device.cpu, false);
    random::uniform_(T, -10, 0, 0);
    T.tag();

    ASSERT_TRUE(T.is_tag());
    ASSERT_TRUE(T.is_braket_form());

    std::vector<UniTensor> Rsvds = linalg::Rsvd_notruncate(T, 1000, true, true, 1, 0, 0., 2, 0);
    ASSERT_EQ(Rsvds.size(), 3);

    const UniTensor& S = Rsvds[0];
    const UniTensor& U = Rsvds[1];
    const UniTensor& vT = Rsvds[2];

    EXPECT_TRUE(S.is_tag());
    EXPECT_TRUE(U.is_tag());
    EXPECT_TRUE(vT.is_tag());
    EXPECT_TRUE(S.is_braket_form());
    EXPECT_TRUE(U.is_braket_form());
    EXPECT_TRUE(vT.is_braket_form());

    EXPECT_EQ(S.bonds()[0].type(), BD_KET);
    EXPECT_EQ(S.bonds()[1].type(), BD_BRA);

    for (int i = 0; i < U.rowrank(); i++) {
      EXPECT_EQ(U.bonds()[i].type(), T.bonds()[i].type());
    }
    EXPECT_EQ(U.bonds().back().type(), BD_BRA);

    EXPECT_EQ(vT.bonds()[0].type(), BD_KET);
    for (int i = 1; i < vT.rank(); i++) {
      EXPECT_EQ(vT.bonds()[i].type(), T.bonds()[T.rowrank() + i - 1].type());
    }

    EXPECT_TRUE(CheckLabels(T, Rsvds));
    EXPECT_TRUE(ReComposeCheck(T, Rsvds));
  }

  /*=====test info=====
  describe:Test Block UniTensor against full Gesvd decomposition.
  input:
    T:Block UniTensor with U1 symmetry.
  ====================*/
  TEST(Rsvd_notruncate, block_u1_compare_gesvd) {
    std::vector<std::string> case_list = {"sym_UT_U1_C128", "sym_UT_U1_F64"};
    for (const auto& case_name : case_list) {
      std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
      fail_msg.Init(test_case_name + ", " + case_name);
      UniTensor src_T = UniTensor::Load(src_data_root + case_name + ".cytnx");
      EXPECT_TRUE(CheckAgainstGesvd(src_T, 1000, 2)) << fail_msg.TraceFailMsgs();
    }
  }

  /*=====test info=====
  describe:Test Block UniTensor against full Gesvd decomposition with mixed symmetry.
  input:
    T:Block UniTensor with U1xZ2 symmetry.
  ====================*/
  TEST(Rsvd_notruncate, block_u1xz2_compare_gesvd) {
    std::vector<std::string> case_list = {"sym_UT_U1xZ2_C128", "sym_UT_U1xZ2_F64"};
    for (const auto& case_name : case_list) {
      std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
      fail_msg.Init(test_case_name + ", " + case_name);
      UniTensor src_T = UniTensor::Load(src_data_root + case_name + ".cytnx");
      EXPECT_TRUE(CheckAgainstGesvd(src_T, 1000, 2)) << fail_msg.TraceFailMsgs();
    }
  }

  /*=====test info=====
  describe:Test BlockFermionic UniTensor against full Gesvd decomposition.
  input:
    T:BlockFermionic UniTensor loaded from test database.
  ====================*/
  TEST(Rsvd_notruncate, block_fermionic_compare_gesvd) {
    std::string test_case_name = UnitTest::GetInstance()->current_test_info()->name();
    fail_msg.Init(test_case_name + ", in-test BlockFermionic tensor");
    UniTensor src_T = BuildCombinedBlockFermionicTensorWithSignflip();
    EXPECT_TRUE(CheckAgainstGesvd(src_T, 1000, 2)) << fail_msg.TraceFailMsgs();
  }

  /*=====test info=====
  describe:Test is_U/is_vT output selection on block and block fermionic tensors.
  input:
    T:Block and BlockFermionic UniTensor.
  ====================*/
  TEST(Rsvd_notruncate, block_and_fermionic_output_size) {
    {
      std::string case_name = "sym_UT_U1_F64";
      UniTensor src_T = UniTensor::Load(src_data_root + case_name + ".cytnx");
      auto out_UV = linalg::Rsvd_notruncate(src_T, 1000, true, true, 1, 0, 0., 2, 0);
      EXPECT_EQ(out_UV.size(), 3) << case_name;
      auto out_S = linalg::Rsvd_notruncate(src_T, 1000, false, false, 1, 0, 0., 2, 0);
      EXPECT_EQ(out_S.size(), 1) << case_name;
      EXPECT_TRUE(SingularValsCorrect(out_S[0], out_UV[0])) << case_name;
    }

    {
      std::string case_name = "constructed_BlockFermionic";
      UniTensor src_T = BuildCombinedBlockFermionicTensorWithSignflip();
      auto out_UV = linalg::Rsvd_notruncate(src_T, 1000, true, true, 1, 0, 0., 2, 0);
      EXPECT_EQ(out_UV.size(), 3) << case_name;
      auto out_S = linalg::Rsvd_notruncate(src_T, 1000, false, false, 1, 0, 0., 2, 0);
      EXPECT_EQ(out_S.size(), 1) << case_name;
      EXPECT_TRUE(SingularValsCorrect(out_S[0], out_UV[0])) << case_name;
    }
  }

  /*=====test info=====
  describe:Block tensor with combined sectors; cover full-kept and truncated sectors.
  input:
    T:U1 block tensor with rowrank=2 where multiple q-index pairs merge to larger sectors.
  ====================*/
  TEST(Rsvd_notruncate, block_combined_sector_full_and_truncated) {
    UniTensor src_T = BuildCombinedBlockTensor();
    std::vector<UniTensor> gesvd = linalg::Gesvd(src_T, true, true);
    const UniTensor& full_S = gesvd[0];

    EXPECT_LT(full_S.Nblocks(), src_T.Nblocks());

    cytnx_uint64 keepdim = FindKeepdimForCategory(src_T, full_S, false, false);
    std::vector<UniTensor> rsvd =
      linalg::Rsvd_notruncate(src_T, keepdim, true, true, 1, 0, 0., 2, 7);
    const auto expected_dims = ExpectedKeptDims(src_T, full_S, keepdim);
    const auto kept_dims = rsvd[0].bonds()[0].getDegeneracies();

    EXPECT_EQ(kept_dims, expected_dims);
    EXPECT_TRUE(CheckPerBlockLeadingSvals(rsvd[0], full_S, 2e-1));

    bool has_full = false, has_trunc = false, has_one_kept = false;
    CheckCategoryCoverage(full_S.bonds()[0].getDegeneracies(), kept_dims, has_full, has_trunc,
                          has_one_kept);
    EXPECT_TRUE(has_trunc);
  }

  /*=====test info=====
  describe:Block tensor with combined sectors; cover one-singular-value-kept sectors.
  input:
    T:U1 block tensor with rowrank=2 where multiple q-index pairs merge to larger sectors.
  ====================*/
  TEST(Rsvd_notruncate, block_combined_sector_one_kept_and_truncated) {
    UniTensor src_T = BuildCombinedBlockTensor();
    std::vector<UniTensor> gesvd = linalg::Gesvd(src_T, true, true);
    const UniTensor& full_S = gesvd[0];

    cytnx_uint64 keepdim = FindKeepdimForCategory(src_T, full_S, false, true);
    std::vector<UniTensor> rsvd =
      linalg::Rsvd_notruncate(src_T, keepdim, true, true, 1, 0, 0., 2, 11);
    const auto expected_dims = ExpectedKeptDims(src_T, full_S, keepdim);
    const auto kept_dims = rsvd[0].bonds()[0].getDegeneracies();

    EXPECT_EQ(kept_dims, expected_dims);
    EXPECT_TRUE(CheckPerBlockLeadingSvals(rsvd[0], full_S, 3e-1));

    bool has_full = false, has_trunc = false, has_one_kept = false;
    CheckCategoryCoverage(full_S.bonds()[0].getDegeneracies(), kept_dims, has_full, has_trunc,
                          has_one_kept);
    EXPECT_TRUE(has_trunc);
    EXPECT_TRUE(has_one_kept);
  }

  /*=====test info=====
  describe:Block tensor with combined U1xZ2 symmetry; cover full-kept and truncated sectors.
  input:
    T:U1xZ2 block tensor with rowrank=2 where multiple q-index pairs merge to larger sectors.
  ====================*/
  TEST(Rsvd_notruncate, block_u1xz2_combined_sector_full_and_truncated) {
    UniTensor src_T = BuildCombinedBlockTensorU1xZ2();
    std::vector<UniTensor> gesvd = linalg::Gesvd(src_T, true, true);
    const UniTensor& full_S = gesvd[0];

    EXPECT_LT(full_S.Nblocks(), src_T.Nblocks());

    cytnx_uint64 keepdim = FindKeepdimForCategory(src_T, full_S, true, false);
    std::vector<UniTensor> rsvd =
      linalg::Rsvd_notruncate(src_T, keepdim, true, true, 1, 0, 0., 2, 19);
    const auto expected_dims = ExpectedKeptDims(src_T, full_S, keepdim);
    const auto kept_dims = rsvd[0].bonds()[0].getDegeneracies();

    EXPECT_EQ(kept_dims, expected_dims);
    EXPECT_TRUE(CheckPerBlockLeadingSvals(rsvd[0], full_S, 2e-1));

    bool has_full = false, has_trunc = false, has_one_kept = false;
    CheckCategoryCoverage(full_S.bonds()[0].getDegeneracies(), kept_dims, has_full, has_trunc,
                          has_one_kept);
    EXPECT_TRUE(has_full);
    EXPECT_TRUE(has_trunc);
  }

  /*=====test info=====
  describe:Block tensor with combined U1xZ2 symmetry; cover one-singular-value-kept sectors.
  input:
    T:U1xZ2 block tensor with rowrank=2 where multiple q-index pairs merge to larger sectors.
  ====================*/
  TEST(Rsvd_notruncate, block_u1xz2_combined_sector_one_kept_and_truncated) {
    UniTensor src_T = BuildCombinedBlockTensorU1xZ2();
    std::vector<UniTensor> gesvd = linalg::Gesvd(src_T, true, true);
    const UniTensor& full_S = gesvd[0];

    cytnx_uint64 keepdim = FindKeepdimForCategory(src_T, full_S, false, true);
    std::vector<UniTensor> rsvd =
      linalg::Rsvd_notruncate(src_T, keepdim, true, true, 1, 0, 0., 2, 23);
    const auto expected_dims = ExpectedKeptDims(src_T, full_S, keepdim);
    const auto kept_dims = rsvd[0].bonds()[0].getDegeneracies();

    EXPECT_EQ(kept_dims, expected_dims);
    EXPECT_TRUE(CheckPerBlockLeadingSvals(rsvd[0], full_S, 3e-1));

    bool has_full = false, has_trunc = false, has_one_kept = false;
    CheckCategoryCoverage(full_S.bonds()[0].getDegeneracies(), kept_dims, has_full, has_trunc,
                          has_one_kept);
    EXPECT_TRUE(has_trunc);
    EXPECT_TRUE(has_one_kept);
  }

  /*=====test info=====
  describe:BlockFermionic tensor with combined sectors; cover full-kept and truncated sectors.
  input:
    T:BlockFermionic tensor with FParity x U1 symmetry and rowrank=2.
  ====================*/
  TEST(Rsvd_notruncate, block_fermionic_combined_sector_full_and_truncated) {
    UniTensor src_T = BuildCombinedBlockFermionicTensorWithSignflip();
    bool has_signflip = false;
    for (auto sf : src_T.signflip()) {
      if (sf) {
        has_signflip = true;
        break;
      }
    }
    EXPECT_TRUE(has_signflip);
    std::vector<UniTensor> gesvd = linalg::Gesvd(src_T, true, true);
    const UniTensor& full_S = gesvd[0];

    EXPECT_LT(full_S.Nblocks(), src_T.Nblocks());

    cytnx_uint64 keepdim = FindKeepdimForCategory(src_T, full_S, true, false);
    std::vector<UniTensor> rsvd =
      linalg::Rsvd_notruncate(src_T, keepdim, true, true, 1, 0, 0., 2, 13);
    const auto expected_dims = ExpectedKeptDims(src_T, full_S, keepdim);
    const auto kept_dims = rsvd[0].bonds()[0].getDegeneracies();

    EXPECT_EQ(kept_dims, expected_dims);
    EXPECT_TRUE(CheckPerBlockLeadingSvals(rsvd[0], full_S, 2e-1));

    bool has_full = false, has_trunc = false, has_one_kept = false;
    CheckCategoryCoverage(full_S.bonds()[0].getDegeneracies(), kept_dims, has_full, has_trunc,
                          has_one_kept);
    EXPECT_TRUE(has_full);
    EXPECT_TRUE(has_trunc);
  }

  /*=====test info=====
  describe:BlockFermionic tensor with combined sectors; cover one-singular-value-kept sectors.
  input:
    T:BlockFermionic tensor with FParity x U1 symmetry and rowrank=2.
  ====================*/
  TEST(Rsvd_notruncate, block_fermionic_combined_sector_one_kept_and_truncated) {
    UniTensor src_T = BuildCombinedBlockFermionicTensorWithSignflip();
    bool has_signflip = false;
    for (auto sf : src_T.signflip()) {
      if (sf) {
        has_signflip = true;
        break;
      }
    }
    EXPECT_TRUE(has_signflip);
    std::vector<UniTensor> gesvd = linalg::Gesvd(src_T, true, true);
    const UniTensor& full_S = gesvd[0];

    cytnx_uint64 keepdim = FindKeepdimForCategory(src_T, full_S, false, true);
    std::vector<UniTensor> rsvd =
      linalg::Rsvd_notruncate(src_T, keepdim, true, true, 1, 0, 0., 2, 17);
    const auto expected_dims = ExpectedKeptDims(src_T, full_S, keepdim);
    const auto kept_dims = rsvd[0].bonds()[0].getDegeneracies();

    EXPECT_EQ(kept_dims, expected_dims);
    EXPECT_TRUE(CheckPerBlockLeadingSvals(rsvd[0], full_S, 3e-1));

    bool has_full = false, has_trunc = false, has_one_kept = false;
    CheckCategoryCoverage(full_S.bonds()[0].getDegeneracies(), kept_dims, has_full, has_trunc,
                          has_one_kept);
    EXPECT_TRUE(has_trunc);
    EXPECT_TRUE(has_one_kept);
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
    T_float.contiguous_();
    ReCompose.permute_(T_float.labels());
    ReCompose.contiguous_();
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

    const double tol = is_double_float_acc ? 1.0e-10 : 1.0e-5;
    return (relative_err < tol);
  }

  bool CheckResult(const std::string& case_name, const cytnx_uint64& keepdim,
                   const cytnx_uint64& power_iteration) {
    // test data source file
    std::string src_file_name = src_data_root + case_name + ".cytnx";
    // answer file
    std::string ans_file_name = ans_data_root + case_name + ".cytnx";
    UniTensor src_T = UniTensor::Load(src_file_name);
    UniTensor ans_T = UniTensor::Load(ans_file_name);  // singular values UniTensor

    // Do Rsvd_notruncate
    std::vector<UniTensor> Rsvds =
      linalg::Rsvd_notruncate(src_T, keepdim, true, true, 1, 0, 0., power_iteration, 0);

    // check labels
    if (!(CheckLabels(src_T, Rsvds))) {
      fail_msg.AppendMsg("The output labels are wrong. ", __func__, __LINE__);
      return false;
    }

    // check answer
    if (!(SingularValsCorrect(Rsvds[0], ans_T))) {
      fail_msg.AppendMsg("The singular values are wrong.. ", __func__, __LINE__);
      return false;
    }

    // check recompose [M - USV*]
    if (!ReComposeCheck(src_T, Rsvds)) {
      fail_msg.AppendMsg(
        "The result is wrong after recomposing. "
        "That's mean T not equal USV* ",
        __func__, __LINE__);
      return false;
    }

    return true;
  }

  bool CheckAgainstGesvd(const UniTensor& src_T, const cytnx_uint64& keepdim,
                         const cytnx_uint64& power_iteration, bool is_U, bool is_vT,
                         cytnx_uint64 mindim, cytnx_uint64 oversampling_summand,
                         double oversampling_factor) {
    std::vector<UniTensor> rsvd =
      linalg::Rsvd_notruncate(src_T, keepdim, is_U, is_vT, mindim, oversampling_summand,
                              oversampling_factor, power_iteration, 0);

    std::vector<UniTensor> gesvd = linalg::Gesvd(src_T, is_U, is_vT);

    if (!SingularValsCorrect(rsvd[0], gesvd[0])) {
      fail_msg.AppendMsg("The singular values differ from Gesvd. ", __func__, __LINE__);
      return false;
    }

    if (is_U || is_vT) {
      if (!CheckLabels(src_T, rsvd)) {
        fail_msg.AppendMsg("The output labels are wrong. ", __func__, __LINE__);
        return false;
      }
    }

    if (is_U && is_vT) {
      if (!ReComposeCheck(src_T, rsvd)) {
        fail_msg.AppendMsg("Recomposition from Rsvd_notruncate is wrong. ", __func__, __LINE__);
        return false;
      }
    }

    return true;
  }

  UniTensor BuildCombinedBlockTensor() {
    std::vector<std::vector<cytnx_int64>> qnums = {{0}, {1}};
    auto syms = std::vector<Symmetry>{Symmetry(SymmetryType::U)};
    Bond l1 = Bond(BD_KET, qnums, {2, 3}, syms);
    Bond l2 = Bond(BD_KET, qnums, {1, 2}, syms);
    Bond r1 = l1.redirect();
    Bond r2 = l2.redirect();

    UniTensor T({l1, l2, r1, r2}, {"l1", "l2", "r1", "r2"}, 2, Type.Double, Device.cpu, false);
    InitUniTensorUniform(T, 23);
    return T;
  }

  UniTensor BuildCombinedBlockTensorU1xZ2() {
    std::vector<std::vector<cytnx_int64>> qnums = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    auto syms = std::vector<Symmetry>{Symmetry(SymmetryType::U), Symmetry(SymmetryType::Z, 2)};
    Bond l1 = Bond(BD_KET, qnums, {1, 2, 1, 2}, syms);
    Bond l2 = Bond(BD_KET, qnums, {2, 1, 2, 1}, syms);
    Bond r1 = l1.redirect();
    Bond r2 = l2.redirect();

    UniTensor T({l1, l2, r1, r2}, {"l1", "l2", "r1", "r2"}, 2, Type.Double, Device.cpu, false);
    InitUniTensorUniform(T, 31);
    return T;
  }

  UniTensor BuildCombinedBlockFermionicTensor() {
    std::vector<std::vector<cytnx_int64>> qnums = {{0, 0}, {1, 1}};
    auto syms = std::vector<Symmetry>{Symmetry::FermionParity(), Symmetry(SymmetryType::U)};
    Bond l1 = Bond(BD_IN, qnums, {2, 3}, syms);
    Bond l2 = Bond(BD_IN, qnums, {1, 2}, syms);
    Bond r1 = l1.redirect();
    Bond r2 = l2.redirect();

    UniTensor T({l1, l2, r1, r2}, {"l1", "l2", "r1", "r2"}, 2, Type.Double, Device.cpu, false);
    InitUniTensorUniform(T, 29);
    return T;
  }

  UniTensor BuildCombinedBlockFermionicTensorWithSignflip() {
    UniTensor base = BuildCombinedBlockFermionicTensor();
    std::vector<std::vector<cytnx_int64>> perms = {
      {1, 0, 2, 3},
      {0, 2, 1, 3},
      {2, 1, 0, 3},
      {0, 1, 3, 2},
    };

    for (const auto& p : perms) {
      UniTensor cand = base.permute(p);
      bool has_signflip = false;
      for (auto sf : cand.signflip()) {
        if (sf) {
          has_signflip = true;
          break;
        }
      }
      if (has_signflip) return cand;
    }
    return base.permute({1, 0, 2, 3});
  }

  std::vector<cytnx_uint64> ExpectedKeptDims(const UniTensor& Tin, const UniTensor& full_svals,
                                             cytnx_uint64 keepdim) {
    const auto full_dims = full_svals.bonds()[0].getDegeneracies();
    cytnx_uint64 rowdim = 1, coldim = 1;
    const auto tshape = Tin.shape();
    for (cytnx_uint64 i = 0; i < Tin.rowrank(); i++) rowdim *= tshape[i];
    for (cytnx_uint64 i = Tin.rowrank(); i < Tin.rank(); i++) coldim *= tshape[i];
    const cytnx_uint64 ten_dim = std::min(rowdim, coldim);

    std::vector<cytnx_uint64> expected(full_dims.size(), 0);
    for (size_t i = 0; i < full_dims.size(); i++) {
      const cytnx_uint64 d = full_dims[i];
      const cytnx_uint64 svalnum = std::max<cytnx_uint64>(
        1, static_cast<cytnx_uint64>(std::ceil(static_cast<double>(keepdim * d) / ten_dim)));
      expected[i] = std::min(d, svalnum);
    }
    return expected;
  }

  bool CheckPerBlockLeadingSvals(const UniTensor& rsvd_svals, const UniTensor& gesvd_svals,
                                 double tol) {
    std::map<std::vector<cytnx_int64>, size_t> gblk_map;
    const auto& gqnums = gesvd_svals.bonds()[0].qnums();
    for (size_t i = 0; i < gqnums.size(); i++) {
      gblk_map[gqnums[i]] = i;
    }

    const auto rblks = rsvd_svals.get_blocks_();
    const auto gblks = gesvd_svals.get_blocks_();
    const auto& rqnums = rsvd_svals.bonds()[0].qnums();
    if (rblks.size() != rqnums.size()) return false;

    for (size_t b = 0; b < rblks.size(); b++) {
      if (gblk_map.find(rqnums[b]) == gblk_map.end()) return false;
      size_t gb = gblk_map[rqnums[b]];
      const auto rdim = rblks[b].shape()[0];
      if (rdim > gblks[gb].shape()[0]) return false;
      for (cytnx_uint64 i = 0; i < rdim; i++) {
        const double r = rblks[b].at<double>({i});
        const double g = gblks[gb].at<double>({i});
        const double denom = std::abs(g) < 1.0 ? 1.0 : std::abs(g);
        if (std::abs(r - g) / denom > tol) return false;
      }
    }
    return true;
  }

  cytnx_uint64 FindKeepdimForCategory(const UniTensor& Tin, const UniTensor& full_svals,
                                      bool require_full, bool require_one_kept) {
    const auto full_dims = full_svals.bonds()[0].getDegeneracies();
    cytnx_uint64 rowdim = 1, coldim = 1;
    const auto tshape = Tin.shape();
    for (cytnx_uint64 i = 0; i < Tin.rowrank(); i++) rowdim *= tshape[i];
    for (cytnx_uint64 i = Tin.rowrank(); i < Tin.rank(); i++) coldim *= tshape[i];
    const cytnx_uint64 ten_dim = std::min(rowdim, coldim);

    for (cytnx_uint64 keepdim = 1; keepdim <= ten_dim; keepdim++) {
      std::vector<cytnx_uint64> kept_dims = ExpectedKeptDims(Tin, full_svals, keepdim);
      bool has_full = false, has_trunc = false, has_one_kept = false;
      CheckCategoryCoverage(full_dims, kept_dims, has_full, has_trunc, has_one_kept);
      if (has_trunc && (!require_full || has_full) && (!require_one_kept || has_one_kept)) {
        return keepdim;
      }
    }
    return std::max<cytnx_uint64>(1, ten_dim / 2);
  }

  void CheckCategoryCoverage(const std::vector<cytnx_uint64>& full_dims,
                             const std::vector<cytnx_uint64>& kept_dims, bool& has_full,
                             bool& has_trunc, bool& has_one_kept) {
    has_full = false;
    has_trunc = false;
    has_one_kept = false;
    for (size_t i = 0; i < full_dims.size(); i++) {
      if (kept_dims[i] == full_dims[i]) has_full = true;
      if (kept_dims[i] < full_dims[i]) has_trunc = true;
      if (kept_dims[i] == 1 && full_dims[i] > 1) has_one_kept = true;
    }
  }
}  // namespace RsvdNoTruncateTest
