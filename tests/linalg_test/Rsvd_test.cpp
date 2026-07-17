#include <gtest/gtest.h>

#include "cytnx.hpp"
#include "test_tools.h"

namespace cytnx {
  namespace {

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
    // normal test

    /*=====test info=====
    describe:Test dense UniTensor only one element.
    input:
      T:Dense UniTensor only one element.
      is_U:true
      is_VT:true
    ====================*/
    TEST(Rsvd, DenseOneElem) {
      std::string test_case_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
      fail_msg.Init(test_case_name);
      int size = 1;
      std::vector<Bond> bonds = {Bond(size), Bond(size), Bond(size)};
      int rowrank = 1;
      bool is_diag = false;
      auto labels = std::vector<std::string>();
      auto T = UniTensor(bonds, labels, rowrank, Type.Double, Device.cpu, is_diag);
      random::uniform_(T, -10, 0, 0);
      std::vector<UniTensor> rsvds = linalg::Rsvd(T, 1);
      EXPECT_TRUE(CheckLabels(T, rsvds)) << fail_msg.TraceFailMsgs();
      EXPECT_TRUE(ReComposeCheck(T, rsvds)) << fail_msg.TraceFailMsgs();
      EXPECT_DOUBLE_EQ(rsvds[0].at<double>({0}), std::abs(T.at<double>({0, 0, 0})))
        << "Singular value is wrong."
        << " line:" << __LINE__ << std::endl;
    }

    TEST(Rsvd, DenseLowRankRectangularAndTransposedExactReconstruction) {
      std::string test_case_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
      fail_msg.Init(test_case_name);

      UniTensor src_T = BuildLowRankRectangularDenseUniTensor(Device.cpu);
      UniTensor src_Tt = src_T.permute({2, 0, 1}, 1).contiguous_();

      CheckLowRankRectangularDenseUniTensorCase(src_T, src_Tt);
    }

    /*=====test info=====
    describe:When keepdim >= #singular values and err=0, nothing is truncated; return_err=1
    returns a scalar zero while return_err>1 returns an empty vector.
    ====================*/
    TEST(Rsvd, TensorNoTruncationReturnsZeroError) {
      Tensor T = Tensor({6, 5}, Type.Double);
      InitTensorUniform(T, 31);
      const cytnx_uint64 full = 5;  // min(6, 5) singular values
      const cytnx_uint64 summand = 0;
      const double factor = 0.;
      const cytnx_uint64 power_it = 2;
      const unsigned int seed = 0;
      std::vector<Tensor> full_ref = linalg::Gesvd(T, true, true);
      for (int return_err : {1, 2}) {
        std::vector<Tensor> out =
          linalg::Rsvd(T, full, 0., true, true, return_err, 1, summand, factor, power_it, seed);
        const std::string label = "return_err=" + std::to_string(return_err);
        CheckTruncatedSvdResult(out, full_ref[0], full, true, true, return_err, 1e-8, label);
        EXPECT_EQ(out[1].shape(), std::vector<cytnx_uint64>({6, full})) << label;
        EXPECT_EQ(out[2].shape(), std::vector<cytnx_uint64>({full, 5})) << label;
      }
    }

    // /*=====test info=====
    // describe:Test Dense UniTensor.
    // input:
    //   T:Dense UniTensor with real or complex real type.
    //   is_U:true
    //   is_VT:true
    // ====================*/
    // TEST(Rsvd, DenseNondiagTest) {
    //   std::vector<std::string> case_list = {"dense_nondiag_C128", "dense_nondiag_F64"};
    //   for (const auto& case_name : case_list) {
    //     std::string test_case_name =
    //     ::testing::UnitTest::GetInstance()->current_test_info()->name();
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
    TEST(Rsvd, ErrDenseDiagTest) {
      int size = 5;
      std::vector<Bond> bonds = {Bond(size), Bond(size)};
      int rowrank = 1;
      bool is_diag = true;
      auto labels = std::vector<std::string>();
      auto T = UniTensor(bonds, labels, rowrank, Type.Double, Device.cpu, is_diag);
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
    TEST(Rsvd, DenseExpSvalsTest) {
      std::vector<std::string> case_list = {"dense_nondiag_exp_Svals_C128",
                                            "dense_nondiag_exp_Svals_F64"};
      for (const auto& case_name : case_list) {
        std::string test_case_name =
          ::testing::UnitTest::GetInstance()->current_test_info()->name();
        fail_msg.Init(test_case_name + ", " + case_name);
        EXPECT_TRUE(CheckResult(case_name, 5, 2)) << fail_msg.TraceFailMsgs();
      }
    }

    /*=====test info=====
    describe:Test Dense UniTensor with exponentially decaying singular values. No power iteration
    in Rsvd. input: T:Dense UniTensor with real or complex real type. is_U:true is_VT:true
    ====================*/
    TEST(Rsvd, DenseExpSvalsNoPowerIterationTest) {
      std::vector<std::string> case_list = {"dense_nondiag_exp_Svals_C128",
                                            "dense_nondiag_exp_Svals_F64"};
      for (const auto& case_name : case_list) {
        std::string test_case_name =
          ::testing::UnitTest::GetInstance()->current_test_info()->name();
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
    TEST(Rsvd, U1ExpSvalsTest) {
      std::vector<std::string> case_list = {"sym_UT_U1_exp_Svals_C128", "sym_UT_U1_exp_Svals_F64"};
      for (const auto& case_name : case_list) {
        std::string test_case_name =
          ::testing::UnitTest::GetInstance()->current_test_info()->name();
        fail_msg.Init(test_case_name + ", " + case_name);
        EXPECT_TRUE(CheckResult(case_name, 5, 2)) << fail_msg.TraceFailMsgs();
      }
    }

    /*=====test info=====
    describe:Test U(1) UniTensor with exponentially decaying singular values and use of
    min_blockdim; input: T:U(1) UniTensor with real or complex real type. is_U:true is_VT:true
    ====================*/
    TEST(Rsvd, U1ExpSvalsMinblockdimTest) {
      std::vector<std::string> case_list = {"sym_UT_U1_exp_Svals_C128", "sym_UT_U1_exp_Svals_F64"};
      for (const auto& case_name : case_list) {
        std::string test_case_name =
          ::testing::UnitTest::GetInstance()->current_test_info()->name();
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
    TEST(Rsvd, BlockFermionicTest) {
      std::string test_case_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
      fail_msg.Init(test_case_name);
      UniTensor src_T = BuildCombinedBlockFermionicTensorWithSignflip();
      std::vector<UniTensor> rsvds = linalg::Rsvd(src_T, 10, 0., true, true, 0, 0, 2, 1, 2, 0);
      EXPECT_TRUE(CheckLabels(src_T, rsvds)) << fail_msg.TraceFailMsgs();
      // Note: For truncated SVD, exact recomposition is not expected due to truncation
      // EXPECT_TRUE(ReComposeCheck(src_T, rsvds)) << fail_msg.TraceFailMsgs();
    }

    /*=====test info=====
    describe:When per-block min_blockdim guarantees already meet or exceed the global keepdim cap
    (keep_dim<=0 internally) but there are further singular values, the return_err path must
    report the actually dropped singular values.
    ====================*/
    TEST(Rsvd, ReturnErrWhenMinBlockdimExhaustsKeepdim) {
      // 2-leg U(1) BlockUT, 3 sectors of size 3x3 each. With min_blockdim=[2,2,2] every block
      // keeps its top 2 SVs and contributes its 3rd to Sall (3 dropped values total);
      // oversampling_summand=2 makes the per-block sampling formula ask for the full 3 SVs per
      // block (the default ceil(keepdim*BlockDim/TenDim)=1 would otherwise leave only 1 SV per
      // block and the min_blockdim cut would never see anything to put into Sall).
      auto syms = std::vector<Symmetry>{Symmetry(SymmetryType::U)};
      Bond bk = Bond(BD_KET, {{0}, {1}, {2}}, {3, 3, 3}, syms);
      UniTensor src_T({bk, bk.redirect()}, {"l", "r"}, 1, Type.Double, Device.cpu, false);
      InitUniTensorUniform(src_T, 23);

      const cytnx_uint64 keepdim = 3;
      const std::vector<cytnx_uint64> min_blockdim = {2, 2, 2};
      const cytnx_uint64 oversampling_summand = 2;

      // return_err = 2: terr holds every dropped singular value (3 total)
      std::vector<UniTensor> rsvd_all = linalg::Rsvd(src_T, keepdim, min_blockdim, 0., true, true,
                                                     2, 1, oversampling_summand, 0., 2, 0);
      Tensor terr_all = rsvd_all.back().get_block_();
      ASSERT_EQ(terr_all.shape(), std::vector<cytnx_uint64>({3}))
        << "terr must hold all 3 dropped singular values, not a 1-element zero";
      double max_abs = 0.0;
      for (cytnx_uint64 i = 0; i < terr_all.storage().size(); ++i)
        max_abs = std::max(max_abs, std::abs(terr_all.storage().at<double>(i)));
      EXPECT_GT(max_abs, 0.0) << "terr values are all zero (previous bug)";

      // return_err = 1: terr is a single element = largest dropped singular value
      std::vector<UniTensor> rsvd_max = linalg::Rsvd(src_T, keepdim, min_blockdim, 0., true, true,
                                                     1, 1, oversampling_summand, 0., 2, 0);
      Tensor terr_max = rsvd_max.back().get_block_();
      ASSERT_TRUE(terr_max.is_scalar());
      EXPECT_NEAR(std::abs(terr_max.item<double>()), max_abs, 1e-10 * (1.0 + max_abs));
    }

    /*=====test info=====
    describe:With err larger than every singular value, the truncation loop would otherwise cut
    down to 1. mindim must restrict the kept count to at least mindim, and the kept values must be
    the mindim largest ones of the full SVD. Uses samplenum = min(m,n) so Rsvd matches a full SVD.
    ====================*/
    TEST(Rsvd, MindimFloor) {
      Tensor T = Tensor({8, 6}, Type.Double);
      InitTensorUniform(T, 19);
      const cytnx_uint64 keepdim = 6;  // == min(m, n) so the randomized projection is full rank
      const cytnx_uint64 floor = 3;
      const cytnx_uint64 oversampling_summand = 0;
      const double oversampling_factor = 0.;
      const cytnx_uint64 power_it = 4;
      std::vector<Tensor> ref = linalg::Gesvd(T, /*is_U=*/true, /*is_vT=*/true);

      std::vector<Tensor> out =
        linalg::Rsvd(T, keepdim, /*err=*/1e9, /*is_U=*/true, /*is_vT=*/true, 0, floor,
                     oversampling_summand, oversampling_factor, power_it, 0);
      ASSERT_EQ(out[0].shape(), std::vector<cytnx_uint64>({floor}));
      for (cytnx_uint64 i = 0; i < floor; ++i) {
        const double got = out[0].astype(Type.Double).storage().at<double>(i);
        const double exp = ref[0].astype(Type.Double).storage().at<double>(i);
        EXPECT_NEAR(got, exp, 1e-8 * (1.0 + std::abs(exp))) << "S[" << i << "]";
      }
    }

    /*=====test info=====
    describe:On a Block UniTensor, with err larger than every singular value, the truncation loop
    would otherwise cut down to 1; mindim must clamp the kept count to at least mindim. Exercises
    the sample_target=max(keepdim,mindim) path on the Block route.
    ====================*/
    TEST(Rsvd, MindimFloorBlockut) {
      auto syms = std::vector<Symmetry>{Symmetry(SymmetryType::U)};
      Bond bk = Bond(BD_KET, {{0}, {1}, {2}}, {3, 3, 3}, syms);
      UniTensor src_T({bk, bk.redirect()}, {"l", "r"}, 1, Type.Double, Device.cpu, false);
      InitUniTensorUniform(src_T, 31);
      const cytnx_uint64 keepdim = 6, mindim = 4;

      std::vector<UniTensor> out =
        linalg::Rsvd(src_T, keepdim, /*err=*/1e9, true, true, 0, mindim, 0, 0., 2, 0);
      cytnx_uint64 total_kept = 0;
      for (auto d : out[0].bonds()[0].getDegeneracies()) total_kept += d;
      EXPECT_GE(total_kept, mindim);
    }

    /*=====test info=====
    describe:Setting min_blockdim must keep at least the requested number of singular values per
    sector. With a small keepdim, the baseline (no min_blockdim) drops entire sectors; with
    min_blockdim={1,1,1} every sector keeps at least one singular value.
    ====================*/
    TEST(Rsvd, MinBlockdimKeepsEachSector) {
      auto syms = std::vector<Symmetry>{Symmetry(SymmetryType::U)};
      Bond bk = Bond(BD_KET, {{0}, {1}, {2}}, {3, 3, 3}, syms);
      UniTensor src_T({bk, bk.redirect()}, {"l", "r"}, 1, Type.Double, Device.cpu, false);
      InitUniTensorUniform(src_T, 29);
      const cytnx_uint64 keepdim = 1;

      // baseline: only the single largest singular value is kept, so most sectors disappear
      std::vector<UniTensor> base = linalg::Rsvd(src_T, keepdim, 0., true, true, 0, 1, 0, 0., 2, 0);
      const auto base_degs = base[0].bonds()[0].getDegeneracies();
      EXPECT_LT(base_degs.size(), 3u)
        << "baseline should drop some sectors; pick a smaller keepdim or larger blocks";

      // with min_blockdim={1,1,1}: every sector must keep at least 1
      std::vector<UniTensor> withf =
        linalg::Rsvd(src_T, keepdim, {1, 1, 1}, 0., true, true, 0, 1, 0, 0., 2, 0);
      const auto withf_degs = withf[0].bonds()[0].getDegeneracies();
      ASSERT_EQ(withf_degs.size(), 3u);
      for (cytnx_uint64 b = 0; b < 3; ++b) EXPECT_GE(withf_degs[b], 1u) << "block " << b;
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

      const double tol = is_double_float_acc ? 1.0e-14 : 1.0e-6;
      return (relative_err < tol);
    }

    UniTensor BuildLowRankRectangularDenseUniTensor(const int device) {
      Tensor left = arange(0, 18, 1, Type.Double, device).reshape({6, 3}) + 1.0;
      Tensor right = arange(0, 15, 1, Type.Double, device).reshape({3, 5}) + 1.0;
      Tensor src = linalg::Matmul(left, right).reshape({2, 3, 5});
      return UniTensor(src, false, 2, {"a", "b", "c"});
    }

    void CheckLowRankRectangularDenseUniTensorCase(const UniTensor& src_T,
                                                   const UniTensor& src_Tt) {
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

    // no use
    void Check_UU_VV_Identity(const UniTensor& Tin, const std::vector<UniTensor>& Tout) {
      const UniTensor& U = Tout[1];
      const UniTensor& V = Tout[2];
      auto UD = U.Dagger();
      UD.relabel_({"0", "1", "9"});
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
      UniTensor rec_T =
        UniTensor::Load(rec_file_name);  // M = U * S * V after correct truncated SVD

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
        fail_msg.AppendMsg("The result is wrong after recomposing, T is not equal to USV*.",
                           __func__, __LINE__);
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
      UniTensor rec_T =
        UniTensor::Load(rec_file_name);  // M = U * S * V after correct truncated SVD

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
        fail_msg.AppendMsg("The result is wrong after recomposing, T is not equal to USV*.",
                           __func__, __LINE__);
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

    // ============================================================================
    // Rsvd no-truncation path: tests below exercise the randomized-SVD pipeline
    // without any post-sketch truncation. Each call sets err=0, return_err=0, and
    // keepdim/oversampling chosen so that all sampled singular values survive.
    // ============================================================================

    static TestFailMsg no_trunc_fail_msg;

    bool CheckNoTruncResult(const std::string& case_name, const cytnx_uint64& keepdim,
                            const cytnx_uint64& power_iteration);
    bool NoTruncReComposeCheck(const UniTensor& Tin, const std::vector<UniTensor>& Tout);
    bool NoTruncCheckLabels(const UniTensor& Tin, const std::vector<UniTensor>& Tout);
    bool NoTruncSingularValsCorrect(const UniTensor& res, const UniTensor& ans);
    bool NoTruncCheckAgainstGesvd(const UniTensor& src_T, const cytnx_uint64& keepdim,
                                  const cytnx_uint64& power_iteration, bool is_U = true,
                                  bool is_vT = true, cytnx_uint64 mindim = 1,
                                  cytnx_uint64 oversampling_summand = 0,
                                  double oversampling_factor = 0.0);
    UniTensor BuildNoTruncCombinedBlockTensor();
    UniTensor BuildNoTruncCombinedBlockTensorU1xZ2();
    UniTensor BuildNoTruncCombinedBlockFermionicTensor();
    UniTensor BuildNoTruncCombinedBlockFermionicTensorWithSignflip();
    std::vector<cytnx_uint64> NoTruncExpectedKeptDims(const UniTensor& Tin,
                                                      const UniTensor& full_svals,
                                                      cytnx_uint64 keepdim);
    bool NoTruncCheckPerBlockLeadingSvals(const UniTensor& rsvd_svals, const UniTensor& gesvd_svals,
                                          double tol);
    cytnx_uint64 NoTruncFindKeepdimForCategory(const UniTensor& Tin, const UniTensor& full_svals,
                                               bool require_full, bool require_one_kept);
    void NoTruncCheckCategoryCoverage(const std::vector<cytnx_uint64>& full_dims,
                                      const std::vector<cytnx_uint64>& kept_dims, bool& has_full,
                                      bool& has_trunc, bool& has_one_kept);
    std::string no_trunc_src_data_root = CYTNX_TEST_DATA_DIR "/common/";
    std::string no_trunc_ans_data_root = CYTNX_TEST_DATA_DIR "/linalg/Rsvd/";
    // normal test

    /*=====test info=====
    describe:Test dense UniTensor only one element.
    input:
      T:Dense UniTensor only one element.
      is_U:true
      is_VT:true
    ====================*/
    TEST(RsvdNoTruncation, DenseOneElem) {
      std::string test_case_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
      no_trunc_fail_msg.Init(test_case_name);
      int size = 1;
      std::vector<Bond> bonds = {Bond(size), Bond(size), Bond(size)};
      int rowrank = 1;
      bool is_diag = false;
      auto labels = std::vector<std::string>();
      auto T = UniTensor(bonds, labels, rowrank, Type.Double, Device.cpu, is_diag);
      random::uniform_(T, -10, 0, 0);
      std::vector<UniTensor> rsvds = linalg::Rsvd(T, 1, 0., true, true, 0, 1, 0, 0., 2);
      EXPECT_TRUE(NoTruncCheckLabels(T, rsvds)) << no_trunc_fail_msg.TraceFailMsgs();
      EXPECT_TRUE(NoTruncReComposeCheck(T, rsvds)) << no_trunc_fail_msg.TraceFailMsgs();
      EXPECT_DOUBLE_EQ(rsvds[0].at<double>({0}), std::abs(T.at<double>({0, 0, 0})))
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
    TEST(RsvdNoTruncation, ErrDenseDiagTest) {
      int size = 5;
      std::vector<Bond> bonds = {Bond(size), Bond(size)};
      int rowrank = 1;
      bool is_diag = true;
      auto labels = std::vector<std::string>();
      auto T = UniTensor(bonds, labels, rowrank, Type.Double, Device.cpu, is_diag);
      random::uniform_(T, 0, 10, 0);
      EXPECT_THROW(
        { std::vector<UniTensor> rsvds = linalg::Rsvd(T, 2, 0., true, true, 0, 1, 0, 0., 2); },
        std::logic_error);
    }

    /*=====test info=====
    describe:error test, Test symmetric diagonal UniTensor.
    input:
      T:Symmetric diagonal UniTensor.
    ====================*/
    TEST(RsvdNoTruncation, ErrSymDiagTest) {
      Bond bond_ket = Bond(BD_KET, {Qs(0), Qs(1), Qs(2)}, {2, 1, 2});
      Bond bond_bra = bond_ket.redirect();
      UniTensor UT = UniTensor({bond_ket, bond_bra}, {}, 1, Type.Double, Device.cpu, true);
      EXPECT_THROW(
        { std::vector<UniTensor> rsvds = linalg::Rsvd(UT, 2, 0., true, true, 0, 1, 0, 0., 2); },
        std::logic_error);
    }

    /*=====test info=====
    describe:error test, Test rank-1 UniTensor.
    input:
      T:rank-1 dense UniTensor.
    ====================*/
    TEST(RsvdNoTruncation, ErrRank1Unitensor) {
      UniTensor T = UniTensor({Bond(8)}, {"x"}, 0, Type.Double, Device.cpu, false);
      random::uniform_(T, 0, 10, 0);
      EXPECT_THROW(
        { std::vector<UniTensor> rsvds = linalg::Rsvd(T, 2, 0., true, true, 0, 1, 0, 0., 2); },
        std::logic_error);
    }

    /*=====test info=====
    describe:Test Dense UniTensor with exponentially decaying singular values.
    input:
      T:Dense UniTensor with real or complex real type.
      is_U:true
      is_VT:true
    ====================*/
    TEST(RsvdNoTruncation, DenseExpSvalsTest) {
      std::vector<std::string> case_list = {"dense_nondiag_exp_Svals_C128",
                                            "dense_nondiag_exp_Svals_F64"};
      for (const auto& case_name : case_list) {
        std::string test_case_name =
          ::testing::UnitTest::GetInstance()->current_test_info()->name();
        no_trunc_fail_msg.Init(test_case_name + ", " + case_name);
        EXPECT_TRUE(CheckNoTruncResult(case_name, 15, 2)) << no_trunc_fail_msg.TraceFailMsgs();
      }
    }

    /*=====test info=====
    describe:Test Dense UniTensor with exponentially decaying singular values. No power iteration
    in Rsvd_notruncate. input: T:Dense UniTensor with real or complex real type. is_U:true
    is_VT:true
    ====================*/
    TEST(RsvdNoTruncation, DenseExpSvalsNoPowerIterationTest) {
      std::vector<std::string> case_list = {"dense_nondiag_exp_Svals_C128",
                                            "dense_nondiag_exp_Svals_F64"};
      for (const auto& case_name : case_list) {
        std::string test_case_name =
          ::testing::UnitTest::GetInstance()->current_test_info()->name();
        no_trunc_fail_msg.Init(test_case_name + ", " + case_name);
        EXPECT_TRUE(CheckNoTruncResult(case_name, 15, 0)) << no_trunc_fail_msg.TraceFailMsgs();
      }
    }

    /*=====test info=====
    describe:Test tagged Dense UniTensor output bond types in Rsvd_notruncate.
    input:
      T:Tagged dense UniTensor with rowrank=2.
      is_U:true
      is_VT:true
    ====================*/
    TEST(RsvdNoTruncation, DenseTaggedOutputBondTypes) {
      std::vector<Bond> bonds = {Bond(2), Bond(3), Bond(4), Bond(5)};
      std::vector<std::string> labels = {"a", "b", "c", "d"};
      UniTensor T(bonds, labels, 2, Type.Double, Device.cpu, false);
      random::uniform_(T, -10, 0, 0);
      T.tag();

      ASSERT_TRUE(T.is_tag());
      ASSERT_TRUE(T.is_braket_form());

      std::vector<UniTensor> rsvds = linalg::Rsvd(T, 1000, 0., true, true, 0, 1, 0, 0., 2, 0);
      ASSERT_EQ(rsvds.size(), 3);

      const UniTensor& S = rsvds[0];
      const UniTensor& U = rsvds[1];
      const UniTensor& vT = rsvds[2];

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

      EXPECT_TRUE(NoTruncCheckLabels(T, rsvds));
      EXPECT_TRUE(NoTruncReComposeCheck(T, rsvds));
    }

    /*=====test info=====
    describe:Test Block UniTensor against full Gesvd decomposition.
    input:
      T:Block UniTensor with U1 symmetry.
    ====================*/
    TEST(RsvdNoTruncation, BlockU1CompareGesvd) {
      std::vector<std::string> case_list = {"sym_UT_U1_C128", "sym_UT_U1_F64"};
      for (const auto& case_name : case_list) {
        std::string test_case_name =
          ::testing::UnitTest::GetInstance()->current_test_info()->name();
        no_trunc_fail_msg.Init(test_case_name + ", " + case_name);
        UniTensor src_T = UniTensor::Load(no_trunc_src_data_root + case_name + ".cytnx");
        EXPECT_TRUE(NoTruncCheckAgainstGesvd(src_T, 1000, 2)) << no_trunc_fail_msg.TraceFailMsgs();
      }
    }

    /*=====test info=====
    describe:Test Block UniTensor against full Gesvd decomposition with mixed symmetry.
    input:
      T:Block UniTensor with U1xZ2 symmetry.
    ====================*/
    TEST(RsvdNoTruncation, BlockU1xz2CompareGesvd) {
      std::vector<std::string> case_list = {"sym_UT_U1xZ2_C128", "sym_UT_U1xZ2_F64"};
      for (const auto& case_name : case_list) {
        std::string test_case_name =
          ::testing::UnitTest::GetInstance()->current_test_info()->name();
        no_trunc_fail_msg.Init(test_case_name + ", " + case_name);
        UniTensor src_T = UniTensor::Load(no_trunc_src_data_root + case_name + ".cytnx");
        EXPECT_TRUE(NoTruncCheckAgainstGesvd(src_T, 1000, 2)) << no_trunc_fail_msg.TraceFailMsgs();
      }
    }

    /*=====test info=====
    describe:Test BlockFermionic UniTensor against full Gesvd decomposition.
    input:
      T:BlockFermionic UniTensor loaded from test database.
    ====================*/
    TEST(RsvdNoTruncation, BlockFermionicCompareGesvd) {
      std::string test_case_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
      no_trunc_fail_msg.Init(test_case_name + ", in-test BlockFermionic tensor");
      UniTensor src_T = BuildNoTruncCombinedBlockFermionicTensorWithSignflip();
      EXPECT_TRUE(NoTruncCheckAgainstGesvd(src_T, 1000, 2)) << no_trunc_fail_msg.TraceFailMsgs();
    }

    /*=====test info=====
    describe:Test is_U/is_vT output selection on block and block fermionic tensors.
    input:
      T:Block and BlockFermionic UniTensor.
    ====================*/
    TEST(RsvdNoTruncation, BlockAndFermionicOutputSize) {
      {
        std::string case_name = "sym_UT_U1_F64";
        UniTensor src_T = UniTensor::Load(no_trunc_src_data_root + case_name + ".cytnx");
        auto out_UV = linalg::Rsvd(src_T, 1000, 0., true, true, 0, 1, 0, 0., 2, 0);
        EXPECT_EQ(out_UV.size(), 3) << case_name;
        auto out_S = linalg::Rsvd(src_T, 1000, 0., false, false, 0, 1, 0, 0., 2, 0);
        EXPECT_EQ(out_S.size(), 1) << case_name;
        EXPECT_TRUE(NoTruncSingularValsCorrect(out_S[0], out_UV[0])) << case_name;
      }

      {
        std::string case_name = "constructed_BlockFermionic";
        UniTensor src_T = BuildNoTruncCombinedBlockFermionicTensorWithSignflip();
        auto out_UV = linalg::Rsvd(src_T, 1000, 0., true, true, 0, 1, 0, 0., 2, 0);
        EXPECT_EQ(out_UV.size(), 3) << case_name;
        auto out_S = linalg::Rsvd(src_T, 1000, 0., false, false, 0, 1, 0, 0., 2, 0);
        EXPECT_EQ(out_S.size(), 1) << case_name;
        EXPECT_TRUE(NoTruncSingularValsCorrect(out_S[0], out_UV[0])) << case_name;
      }
    }

    bool NoTruncReComposeCheck(const UniTensor& Tin, const std::vector<UniTensor>& Tout) {
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
      T_float.contiguous_();
      recomposed.permute_(T_float.labels());
      recomposed.contiguous_();
      bool is_eq = AreNearlyEqUniTensor(T_float, recomposed, tol);
      return is_eq;
    }

    bool NoTruncCheckLabels(const UniTensor& Tin, const std::vector<UniTensor>& Tout) {
      const std::vector<std::string>& in_labels = Tin.labels();
      const std::vector<std::string>& s_labels = Tout[0].labels();
      const std::vector<std::string>& u_labels = Tout[1].labels();
      const std::vector<std::string>& v_labels = Tout[2].labels();
      // check S
      if (s_labels[0] != "_aux_L") {
        no_trunc_fail_msg.AppendMsg("The label of the left leg in 'S' is wrong. ", __func__,
                                    __LINE__);
        return false;
      }
      if (s_labels[1] != "_aux_R") {
        no_trunc_fail_msg.AppendMsg("The label of the left leg in 'S' is wrong. ", __func__,
                                    __LINE__);
        return false;
      }
      // check U
      for (size_t i = 0; i < u_labels.size() - 1; ++i) {  // exclude U final lags label
        if (u_labels[i] != in_labels[i]) {
          no_trunc_fail_msg.AppendMsg("The label of 'U' is wrong. ", __func__, __LINE__);
          return false;
        }
      }
      // check V
      for (size_t i = 1; i < v_labels.size(); ++i) {  // exclude VT first lags label
        auto in_indx = u_labels.size() - 2 + i;
        if (v_labels[i] != in_labels[in_indx]) {
          no_trunc_fail_msg.AppendMsg("The label of 'V' is wrong. ", __func__, __LINE__);
          return false;
        }
      }
      return true;
    }

    bool NoTruncSingularValsCorrect(const UniTensor& res, const UniTensor& ans) {
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

    bool CheckNoTruncResult(const std::string& case_name, const cytnx_uint64& keepdim,
                            const cytnx_uint64& power_iteration) {
      // test data source file
      std::string src_file_name = no_trunc_src_data_root + case_name + ".cytnx";
      // answer file
      std::string ans_file_name = no_trunc_ans_data_root + case_name + ".cytnx";
      UniTensor src_T = UniTensor::Load(src_file_name);
      UniTensor ans_T = UniTensor::Load(ans_file_name);  // singular values UniTensor

      // Do Rsvd_notruncate
      std::vector<UniTensor> rsvds =
        linalg::Rsvd(src_T, keepdim, 0., true, true, 0, 1, 0, 0., power_iteration, 0);

      // check labels
      if (!NoTruncCheckLabels(src_T, rsvds)) {
        no_trunc_fail_msg.AppendMsg("The output labels are wrong. ", __func__, __LINE__);
        return false;
      }

      // check answer
      if (!NoTruncSingularValsCorrect(rsvds[0], ans_T)) {
        std::cout << rsvds[0] << std::endl;
        std::cout << ans_T << std::endl;
        no_trunc_fail_msg.AppendMsg("The singular values are wrong. ", __func__, __LINE__);
        return false;
      }

      // check recompose [M - USV*]
      if (!NoTruncReComposeCheck(src_T, rsvds)) {
        no_trunc_fail_msg.AppendMsg(
          "The result is wrong after recomposing, T is not equal to USV*.", __func__, __LINE__);
        return false;
      }

      return true;
    }

    bool NoTruncCheckAgainstGesvd(const UniTensor& src_T, const cytnx_uint64& keepdim,
                                  const cytnx_uint64& power_iteration, bool is_U, bool is_vT,
                                  cytnx_uint64 mindim, cytnx_uint64 oversampling_summand,
                                  double oversampling_factor) {
      std::vector<UniTensor> rsvd =
        linalg::Rsvd(src_T, keepdim, 0., is_U, is_vT, 0, mindim, oversampling_summand,
                     oversampling_factor, power_iteration, 0);

      std::vector<UniTensor> gesvd = linalg::Gesvd(src_T, is_U, is_vT);

      if (!NoTruncSingularValsCorrect(rsvd[0], gesvd[0])) {
        std::cout << rsvd[0] << std::endl;
        std::cout << gesvd[0] << std::endl;
        no_trunc_fail_msg.AppendMsg("The singular values differ from Gesvd. ", __func__, __LINE__);
        return false;
      }

      if (is_U || is_vT) {
        if (!NoTruncCheckLabels(src_T, rsvd)) {
          no_trunc_fail_msg.AppendMsg("The output labels are wrong. ", __func__, __LINE__);
          return false;
        }
      }

      if (is_U && is_vT) {
        if (!NoTruncReComposeCheck(src_T, rsvd)) {
          no_trunc_fail_msg.AppendMsg("Recomposition from Rsvd_notruncate is wrong. ", __func__,
                                      __LINE__);
          return false;
        }
      }

      return true;
    }

    UniTensor BuildNoTruncCombinedBlockTensor() {
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

    UniTensor BuildNoTruncCombinedBlockTensorU1xZ2() {
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

    UniTensor BuildNoTruncCombinedBlockFermionicTensor() {
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

    UniTensor BuildNoTruncCombinedBlockFermionicTensorWithSignflip() {
      UniTensor base = BuildNoTruncCombinedBlockFermionicTensor();
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

    std::vector<cytnx_uint64> NoTruncExpectedKeptDims(const UniTensor& Tin,
                                                      const UniTensor& full_svals,
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

    bool NoTruncCheckPerBlockLeadingSvals(const UniTensor& rsvd_svals, const UniTensor& gesvd_svals,
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

    cytnx_uint64 NoTruncFindKeepdimForCategory(const UniTensor& Tin, const UniTensor& full_svals,
                                               bool require_full, bool require_one_kept) {
      const auto full_dims = full_svals.bonds()[0].getDegeneracies();
      cytnx_uint64 rowdim = 1, coldim = 1;
      const auto tshape = Tin.shape();
      for (cytnx_uint64 i = 0; i < Tin.rowrank(); i++) rowdim *= tshape[i];
      for (cytnx_uint64 i = Tin.rowrank(); i < Tin.rank(); i++) coldim *= tshape[i];
      const cytnx_uint64 ten_dim = std::min(rowdim, coldim);

      for (cytnx_uint64 keepdim = 1; keepdim <= ten_dim; keepdim++) {
        std::vector<cytnx_uint64> kept_dims = NoTruncExpectedKeptDims(Tin, full_svals, keepdim);
        bool has_full = false, has_trunc = false, has_one_kept = false;
        NoTruncCheckCategoryCoverage(full_dims, kept_dims, has_full, has_trunc, has_one_kept);
        if (has_trunc && (!require_full || has_full) && (!require_one_kept || has_one_kept)) {
          return keepdim;
        }
      }
      return std::max<cytnx_uint64>(1, ten_dim / 2);
    }

    void NoTruncCheckCategoryCoverage(const std::vector<cytnx_uint64>& full_dims,
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
  }  // namespace
}  // namespace cytnx
