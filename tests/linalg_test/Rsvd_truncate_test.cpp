#include <gtest/gtest.h>

#include "../test_tools.h"
#include "cytnx.hpp"

namespace cytnx {
  namespace test {
    namespace {

      static TestFailMsg fail_msg;

      static bool CheckResult(const std::string& case_name, const cytnx_uint64& keepdim,
                              const cytnx_uint64& power_iteration);
      static bool ReComposeCheck(const UniTensor& Tin, const std::vector<UniTensor>& Tout);
      static bool CheckLabels(const UniTensor& Tin, const std::vector<UniTensor>& Tout);
      static bool SingularValsCorrect(const UniTensor& res, const UniTensor& ans);
      static std::string src_data_root = CYTNX_TEST_DATA_DIR "/common/";
      static std::string ans_data_root = CYTNX_TEST_DATA_DIR "/linalg/Svd_truncate/";
      // normal test

      /*=====test info=====
      describe:Test dense UniTensor only one element.
      input:
        T:Dense UniTensor only one element.
        is_U:true
        is_VT:true
      ====================*/
      TEST(RsvdTruncate, DenseOneElem) {
        std::string test_case_name =
          ::testing::UnitTest::GetInstance()->current_test_info()->name();
        fail_msg.Init(test_case_name);
        int size = 1;
        std::vector<Bond> bonds = {Bond(size), Bond(size), Bond(size)};
        int rowrank = 1;
        bool is_diag = false;
        auto labels = std::vector<std::string>();
        auto T = UniTensor(bonds, labels, rowrank, Type.Double, Device.cpu, is_diag);
        random::uniform_(T, -10, 0, 0);
        std::vector<UniTensor> Rsvds = linalg::Rsvd_truncate(T, 1);
        EXPECT_TRUE(CheckLabels(T, Rsvds)) << fail_msg.TraceFailMsgs();
        EXPECT_TRUE(ReComposeCheck(T, Rsvds)) << fail_msg.TraceFailMsgs();
        EXPECT_EQ(Rsvds[0].at<double>({0}), std::abs(T.at<double>({0, 0, 0})))
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
      // TEST(RsvdTruncate, DenseNondiagTest) {
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
      TEST(RsvdTruncate, ErrDenseDiagTest) {
        int size = 5;
        std::vector<Bond> bonds = {Bond(size), Bond(size)};
        int rowrank = 1;
        bool is_diag = true;
        auto labels = std::vector<std::string>();
        auto T = UniTensor(bonds, labels, rowrank, Type.Double, Device.cpu, is_diag);
        random::uniform_(T, 0, 10, 0);
        EXPECT_THROW({ std::vector<UniTensor> Rsvds = linalg::Rsvd_truncate(T, 2); },
                     std::logic_error);
      }

      /*=====test info=====
      describe:Test Dense UniTensor with exponentially decaying singular values.
      input:
        T:Dense UniTensor with real or complex real type.
        is_U:true
        is_VT:true
      ====================*/
      TEST(RsvdTruncate, DenseExpSvalsTest) {
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
      TEST(RsvdTruncate, DenseExpSvalsNoPowerIterationTest) {
        std::vector<std::string> case_list = {"dense_nondiag_exp_Svals_C128",
                                              "dense_nondiag_exp_Svals_F64"};
        for (const auto& case_name : case_list) {
          std::string test_case_name =
            ::testing::UnitTest::GetInstance()->current_test_info()->name();
          fail_msg.Init(test_case_name + ", " + case_name);
          EXPECT_TRUE(CheckResult(case_name, 5, 0)) << fail_msg.TraceFailMsgs();
        }
      }

      static bool ReComposeCheck(const UniTensor& Tin, const std::vector<UniTensor>& Tout) {
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
        bool is_eq = AreNearlyEqUniTensor(T_float, ReCompose.permute_(T_float.labels()), tol);
        return is_eq;
      }

      static bool CheckLabels(const UniTensor& Tin, const std::vector<UniTensor>& Tout) {
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

      static bool SingularValsCorrect(const UniTensor& res, const UniTensor& ans) {
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
      static void Check_UU_VV_Identity(const UniTensor& Tin, const std::vector<UniTensor>& Tout) {
        const UniTensor& U = Tout[1];
        const UniTensor& V = Tout[2];
        auto UD = U.Dagger();
        UD.relabel_({"0", "1", "9"});
        UD.permute_({2, 0, 1}, 1);
        auto UUD = Contract(U, UD);
      }

      static bool CheckResult(const std::string& case_name, const cytnx_uint64& keepdim,
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

        // Do Rsvd_truncate
        std::vector<UniTensor> Rsvds =
          linalg::Rsvd_truncate(src_T, keepdim, 0, true, true, 0, 0, 2, 1, power_iteration, 0);

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
        if (!ReComposeCheck(rec_T, Rsvds)) {
          fail_msg.AppendMsg(
            "The result is wrong after recomposing. "
            "That's mean T not equal USV* ",
            __func__, __LINE__);
          return false;
        }

        return true;
      }

      /*=====test info=====
      describe:Tensor-level Rsvd_truncate must work for all (is_U, is_vT) combinations.
      With a fixed seed the random isometry is identical across flag choices, so the kept
      singular values (and any requested matrices) must match the all-true reference.
      ====================*/
      TEST(RsvdTruncate, FlagCombinationsDense) {
        const unsigned int seed = 42;
        const cytnx_uint64 keep = 3;
        const cytnx_uint64 summand = 3;  // samplenum = 6 == min(8,6), so Rsvd matches a full SVD
        const double factor = 0.;
        const cytnx_uint64 power_it = 4;

        for (auto dtype : {Type.Double, Type.ComplexDouble}) {
          Tensor T = Tensor({8, 6}, dtype);
          InitTensorUniform(T, /*seed=*/3);

          // independent reference for singular-value content: full Gesvd. With samplenum = min(m,n)
          // and a few power iterations, Rsvd's S values agree with Gesvd's.
          std::vector<Tensor> gesvd_ref = linalg::Gesvd(T, true, true);

          // Rsvd reference for U/vT (these depend on the random isometry; same seed -> same basis).
          std::vector<Tensor> ref =
            linalg::Rsvd_truncate(T, keep, 0., true, true, 0, 1, summand, factor, power_it, seed);

          for (bool is_U : {false, true}) {
            for (bool is_vT : {false, true}) {
              for (int return_err : {0, 1, 2}) {
                std::vector<Tensor> out = linalg::Rsvd_truncate(
                  T, keep, 0., is_U, is_vT, return_err, 1, summand, factor, power_it, seed);
                const std::string label = "is_U=" + std::to_string(is_U) +
                                          " is_vT=" + std::to_string(is_vT) +
                                          " return_err=" + std::to_string(return_err);
                CheckTruncatedSvdResult(out, gesvd_ref[0], keep, is_U, is_vT, return_err, 1e-8,
                                        label);

                cytnx_uint64 idx = 1;
                if (is_U) {
                  EXPECT_EQ(out[idx].shape()[1], keep) << label;  // U truncated to keep columns
                  EXPECT_TRUE(AreNearlyEqTensor(out[idx], ref[1], 1e-10)) << label;
                  ++idx;
                }
                if (is_vT) {
                  EXPECT_EQ(out[idx].shape()[0], keep) << label;  // vT truncated to keep rows
                  EXPECT_TRUE(AreNearlyEqTensor(out[idx], ref[2], 1e-10)) << label;
                }
              }
            }
          }
        }
      }

    }  // namespace
  }  // namespace test
}  // namespace cytnx
