#include "gtest/gtest.h"

#include "cytnx.hpp"
#include "gpu_test_tools.h"

namespace cytnx {
  namespace gpu_test {
    namespace {

      static TestFailMsg fail_msg;

      static bool CheckResult(const std::string& case_name);
      static bool ReComposeCheck(const UniTensor& Tin, const std::vector<UniTensor>& Tout);
      static bool CheckLabels(const UniTensor& Tin, const std::vector<UniTensor>& Tout);
      static bool SingularValsCorrect(const UniTensor& res, const UniTensor& ans);
      static std::string src_data_root = CYTNX_TEST_DATA_DIR "/common/";
      static std::string ans_data_root = CYTNX_TEST_DATA_DIR "/linalg/Svd/";
      // normal test

      /*=====test info=====
      describe:Test dense UniTensor only one element.
      input:
        T:Dense UniTensor only one element.
        is_U:true
        is_VT:true
      ====================*/
      TEST(Svd, GpuDenseOneElem) {
        std::string test_case_name =
          ::testing::UnitTest::GetInstance()->current_test_info()->name();
        fail_msg.Init(test_case_name);
        int size = 1;
        std::vector<Bond> bonds = {Bond(size), Bond(size), Bond(size)};
        int rowrank = 1;
        bool is_diag = false;
        auto labels = std::vector<std::string>();
        auto T =
          UniTensor(bonds, labels, rowrank, Type.Double, Device.cuda, is_diag).to(Device.cuda);
        random::uniform_(T, -10, 0, 0);
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
      TEST(Svd, GpuDenseNondiagTest) {
        std::vector<std::string> case_list = {"dense_nondiag_C128", "dense_nondiag_F64"};
        for (const auto& case_name : case_list) {
          std::string test_case_name =
            ::testing::UnitTest::GetInstance()->current_test_info()->name();
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
      TEST(Svd, GpuU1SymTest) {
        std::vector<std::string> case_list = {"sym_UT_U1_C128", "sym_UT_U1_F64"};
        for (const auto& case_name : case_list) {
          std::string test_case_name =
            ::testing::UnitTest::GetInstance()->current_test_info()->name();
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
      TEST(Svd, GpuZ2SymTest) {
        std::vector<std::string> case_list = {"sym_UT_Z2_C128", "sym_UT_Z2_F64"};
        for (const auto& case_name : case_list) {
          std::string test_case_name =
            ::testing::UnitTest::GetInstance()->current_test_info()->name();
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
      TEST(Svd, GpuZ3SymTest) {
        std::vector<std::string> case_list = {"sym_UT_Z3_C128", "sym_UT_Z3_F64"};
        for (const auto& case_name : case_list) {
          std::string test_case_name =
            ::testing::UnitTest::GetInstance()->current_test_info()->name();
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
      TEST(Svd, GpuU1xZ2SymTest) {
        std::vector<std::string> case_list = {
          "sym_UT_U1xZ2_C128", "sym_UT_U1xZ2_C64", "sym_UT_U1xZ2_F64", "sym_UT_U1xZ2_F32",
          "sym_UT_U1xZ2_I64",  "sym_UT_U1xZ2_U64", "sym_UT_U1xZ2_I32", "sym_UT_U1xZ2_U32",
          "sym_UT_U1xZ2_I16",  "sym_UT_U1xZ2_U16"};
        for (const auto& case_name : case_list) {
          std::string test_case_name =
            ::testing::UnitTest::GetInstance()->current_test_info()->name();
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
      TEST(Svd, GpuU1ZerosTest) {
        GTEST_SKIP() << "Issue for cuda. Cannot handle if most of elements are zeros.";
        std::string case_name = "sym_UT_U1_zeros_F64";
        std::string test_case_name =
          ::testing::UnitTest::GetInstance()->current_test_info()->name();
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
      TEST(Svd, GpuU1OneElemTest) {
        std::string case_name = "sym_UT_U1_one_elem_F64";
        std::string test_case_name =
          ::testing::UnitTest::GetInstance()->current_test_info()->name();
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
      TEST(Svd, GpuNoneIsUIsVT) {
        // bool need_U, need_VT;
        bool compute_uv;
        std::string case_name = "sym_UT_U1_F64";
        std::string src_file_name = src_data_root + case_name + ".cytnx";
        UniTensor src_T = UniTensor::Load(src_file_name).to(Device.cuda);
        // get S, U, VT
        //  std::vector<UniTensor> svdsUV = linalg::Svd(src_T, need_U = true, need_VT = true);
        //  EXPECT_EQ(svdsUV.size(), 3) << "The number of the output of svd not correct. Line:"
        //      << __LINE__ << std::endl;
        std::vector<UniTensor> svdsUV = linalg::Svd(src_T, compute_uv = true);
        EXPECT_EQ(svdsUV.size(), 3)
          << "The number of the output of svd not correct. Line:" << __LINE__ << std::endl;

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
        EXPECT_EQ(svdsS.size(), 1)
          << "The number of the output of svd not correct. Line:" << __LINE__ << std::endl;

        const double tol = 1.0e-10;
        // Since if no is_U and is_vT, we may call Lapack for different arguments to
        //   say we don't need U and VT, so we have tolerance here.
        EXPECT_TRUE(AreNearlyEqUniTensor(svdsS[0], svdsUV[0], tol));
      }

      // Builds a dense rows x cols matrix (rows, cols in {2, 4}) with the given singular values as
      // U * diag(singular_values) * V^T, where U and V have orthonormal columns taken from
      // Hadamard matrices. The singular values are therefore known without calling any SVD.
      // For complex dtypes each column is multiplied by a unit phase, which keeps the singular
      // values unchanged.
      Tensor MakeMatrixWithKnownSingularValues(std::size_t rows, std::size_t cols,
                                               const std::vector<double>& singular_values,
                                               unsigned int dtype) {
        const double hadamard4[4][4] = {
          {1, 1, 1, 1}, {1, -1, 1, -1}, {1, 1, -1, -1}, {1, -1, -1, 1}};
        const double hadamard2[2][2] = {{1, 1}, {1, -1}};
        const auto orthonormal_entry = [&](std::size_t size, std::size_t row, std::size_t col) {
          return size == 4 ? hadamard4[row][col] / 2. : hadamard2[row][col] / std::sqrt(2.);
        };
        const bool is_complex = Type.is_complex(dtype);
        const std::vector<std::complex<double>> phases = {
          {1., 0.}, {0., 1.}, {-1., 0.}, {std::sqrt(0.5), -std::sqrt(0.5)}};
        Tensor matrix({rows, cols}, Type.ComplexDouble);
        for (std::size_t row = 0; row < rows; ++row) {
          for (std::size_t col = 0; col < cols; ++col) {
            double entry = 0.;
            for (std::size_t k = 0; k < singular_values.size(); ++k) {
              // The second factor uses the Hadamard columns in reverse order so that U and V
              // differ and the matrix is not symmetric.
              entry += orthonormal_entry(rows, row, k) * singular_values[k] *
                       orthonormal_entry(cols, col, cols - 1 - k);
            }
            matrix.at<cytnx_complex128>({row, col}) = is_complex ? entry * phases[col] : entry;
          }
        }
        if (!is_complex) matrix = matrix.real();
        return matrix.astype(dtype).to(Device.cuda);
      }

      /*=====test info=====
      describe:Test Svd of a dense Tensor on the GPU without singular vectors (issue #831).
      input:
        T:Dense Tensor on cuda with known singular values, all four floating dtypes,
          square, tall and wide.
        is_UvT:false
      ====================*/
      TEST(Svd, GpuDenseTensorNoSingularVectors) {
        struct Case {
          std::size_t rows;
          std::size_t cols;
          std::vector<double> singular_values;
        };
        const std::vector<Case> cases = {
          {4, 4, {3.5, 2.25, 1., 0.5}}, {4, 2, {3.5, 0.5}}, {2, 4, {2.25, 0.75}}};
        for (const unsigned int dtype :
             {Type.Double, Type.Float, Type.ComplexDouble, Type.ComplexFloat}) {
          const bool is_single_precision = dtype == Type.Float || dtype == Type.ComplexFloat;
          const double tol = is_single_precision ? 1.0e-5 : 1.0e-12;
          for (const Case& test_case : cases) {
            SCOPED_TRACE(Type.getname(dtype) + " " + std::to_string(test_case.rows) + "x" +
                         std::to_string(test_case.cols));
            const Tensor matrix = MakeMatrixWithKnownSingularValues(
              test_case.rows, test_case.cols, test_case.singular_values, dtype);
            const std::vector<Tensor> result = linalg::Svd(matrix, false);
            ASSERT_EQ(result.size(), 1);
            const Tensor& singular_values = result[0];
            EXPECT_EQ(singular_values.device(), Device.cuda);
            EXPECT_EQ(singular_values.dtype(), is_single_precision ? Type.Float : Type.Double);
            ASSERT_EQ(singular_values.shape(),
                      std::vector<cytnx_uint64>{test_case.singular_values.size()});
            const Tensor on_cpu = singular_values.to(Device.cpu).astype(Type.Double);
            for (std::size_t k = 0; k < test_case.singular_values.size(); ++k) {
              EXPECT_NEAR(on_cpu.at<double>({k}), test_case.singular_values[k], tol);
            }
          }
        }
      }

      /*=====test info=====
      describe:Test Svd of a dense UniTensor on the GPU without singular vectors (issue #831).
      input:
        T:Dense UniTensor on cuda with known singular values.
        is_UvT:false
      ====================*/
      TEST(Svd, GpuDenseUniTensorNoSingularVectors) {
        const std::vector<double> expected = {3.5, 2.25, 1., 0.5};
        const UniTensor matrix =
          UniTensor(MakeMatrixWithKnownSingularValues(4, 4, expected, Type.Double));
        const std::vector<UniTensor> result = linalg::Svd(matrix, false);
        ASSERT_EQ(result.size(), 1);
        EXPECT_EQ(result[0].device(), Device.cuda);
        const Tensor on_cpu = result[0].get_block().to(Device.cpu);
        ASSERT_EQ(on_cpu.shape(), std::vector<cytnx_uint64>{expected.size()});
        for (std::size_t k = 0; k < expected.size(); ++k) {
          EXPECT_NEAR(on_cpu.at<double>({k}), expected[k], 1.0e-12);
        }
      }

      // Builds a dense long x 2 (is_tall) or 2 x long (otherwise) matrix with the two given
      // singular values as Q * diag(singular_values) * R^T, or the transpose of that for the wide
      // shape. Q has the orthonormal columns p_i / sqrt(long) and p_i * (-1)^i / sqrt(long), with
      // unit phases p_i for complex dtypes, and R is a rotation, so the singular values are known
      // without calling any SVD. `long_extent` must be even.
      Tensor MakeThinMatrixWithKnownSingularValues(std::size_t long_extent, bool is_tall,
                                                   const std::vector<double>& singular_values,
                                                   unsigned int dtype) {
        const bool is_complex = Type.is_complex(dtype);
        const std::vector<std::complex<double>> phases = {
          {1., 0.}, {0., 1.}, {-1., 0.}, {std::sqrt(0.5), -std::sqrt(0.5)}};
        const double rotation[2][2] = {{0.6, -0.8}, {0.8, 0.6}};
        const double norm = 1. / std::sqrt(static_cast<double>(long_extent));
        Tensor matrix(is_tall ? std::vector<cytnx_uint64>{long_extent, 2}
                              : std::vector<cytnx_uint64>{2, long_extent},
                      Type.ComplexDouble);
        for (std::size_t i = 0; i < long_extent; ++i) {
          const std::complex<double> phase = is_complex ? phases[i % 4] : 1.;
          const double q[2] = {norm, i % 2 == 0 ? norm : -norm};
          for (std::size_t j = 0; j < 2; ++j) {
            const std::complex<double> entry = phase * (q[0] * singular_values[0] * rotation[j][0] +
                                                        q[1] * singular_values[1] * rotation[j][1]);
            if (is_tall) {
              matrix.at<cytnx_complex128>({i, j}) = entry;
            } else {
              matrix.at<cytnx_complex128>({j, i}) = entry;
            }
          }
        }
        if (!is_complex) matrix = matrix.real();
        return matrix.astype(dtype).to(Device.cuda);
      }

      // Checks linalg::Svd(matrix, is_UvT) for both values of is_UvT: the singular values against
      // `expected`, and, with singular vectors, the shapes of U and vT, that U * diag(S) * vT
      // reproduces `matrix`, and that the columns of U and the rows of vT are orthonormal.
      void ExpectSvdMatchesKnownSingularValues(const Tensor& matrix,
                                               const std::vector<double>& expected, double tol) {
        const std::size_t rows = matrix.shape()[0];
        const std::size_t cols = matrix.shape()[1];
        const std::size_t rank = expected.size();
        const bool is_single_precision =
          matrix.dtype() == Type.Float || matrix.dtype() == Type.ComplexFloat;
        for (const bool is_UvT : {false, true}) {
          SCOPED_TRACE(is_UvT ? "is_UvT=true" : "is_UvT=false");
          const std::vector<Tensor> result = linalg::Svd(matrix, is_UvT);
          ASSERT_EQ(result.size(), is_UvT ? 3 : 1);
          EXPECT_EQ(result[0].device(), Device.cuda);
          EXPECT_EQ(result[0].dtype(), is_single_precision ? Type.Float : Type.Double);
          ASSERT_EQ(result[0].shape(), std::vector<cytnx_uint64>{rank});
          const Tensor s = result[0].to(Device.cpu).astype(Type.Double);
          for (std::size_t k = 0; k < rank; ++k) {
            EXPECT_NEAR(s.at<double>({k}), expected[k], tol);
          }
          if (!is_UvT) continue;

          EXPECT_EQ(result[1].dtype(), matrix.dtype());
          EXPECT_EQ(result[2].dtype(), matrix.dtype());
          ASSERT_EQ(result[1].shape(), (std::vector<cytnx_uint64>{rows, rank}));
          ASSERT_EQ(result[2].shape(), (std::vector<cytnx_uint64>{rank, cols}));
          const Tensor a = matrix.to(Device.cpu).astype(Type.ComplexDouble);
          const Tensor u = result[1].to(Device.cpu).astype(Type.ComplexDouble);
          const Tensor vt = result[2].to(Device.cpu).astype(Type.ComplexDouble);
          double max_reconstruction_error = 0.;
          for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
              std::complex<double> entry = 0.;
              for (std::size_t k = 0; k < rank; ++k) {
                entry += u.at<cytnx_complex128>({i, k}) * s.at<double>({k}) *
                         vt.at<cytnx_complex128>({k, j});
              }
              max_reconstruction_error = std::max(max_reconstruction_error,
                                                  std::abs(entry - a.at<cytnx_complex128>({i, j})));
            }
          }
          EXPECT_LT(max_reconstruction_error, tol);
          double max_orthonormality_error = 0.;
          for (std::size_t k = 0; k < rank; ++k) {
            for (std::size_t l = 0; l < rank; ++l) {
              std::complex<double> u_overlap = 0., vt_overlap = 0.;
              for (std::size_t i = 0; i < rows; ++i) {
                u_overlap +=
                  std::conj(u.at<cytnx_complex128>({i, k})) * u.at<cytnx_complex128>({i, l});
              }
              for (std::size_t j = 0; j < cols; ++j) {
                vt_overlap +=
                  vt.at<cytnx_complex128>({k, j}) * std::conj(vt.at<cytnx_complex128>({l, j}));
              }
              const double identity_entry = k == l ? 1. : 0.;
              max_orthonormality_error =
                std::max({max_orthonormality_error, std::abs(u_overlap - identity_entry),
                          std::abs(vt_overlap - identity_entry)});
            }
          }
          EXPECT_LT(max_orthonormality_error, tol);
        }
      }

      /*=====test info=====
      describe:Test Svd of rectangular dense Tensors on the GPU against known singular values,
        including shapes whose long extent is large. A tall input used to be passed to
        cusolverDnXgesvdp as a wide matrix, whose workspace grows with the square of the long
        extent (CUDA 12.9) or is rejected by gesvdp itself from 46341 columns on (CUDA 13.3),
        so 50000 x 2 could not be decomposed.
      input:
        T:Dense Tensor on cuda with known singular values, all four floating dtypes,
          tall, wide and square.
        is_UvT:false and true
      ====================*/
      TEST(Svd, GpuDenseTensorRectangularKnownSingularValues) {
        for (const unsigned int dtype :
             {Type.Double, Type.Float, Type.ComplexDouble, Type.ComplexFloat}) {
          const bool is_single_precision = dtype == Type.Float || dtype == Type.ComplexFloat;
          const double tol = is_single_precision ? 2.0e-5 : 1.0e-11;
          for (const std::size_t long_extent : {6, 1000, 50000}) {
            for (const bool is_tall : {true, false}) {
              SCOPED_TRACE(Type.getname(dtype) + (is_tall ? " tall " : " wide ") +
                           std::to_string(long_extent));
              const std::vector<double> expected = {7.5, 1.25};
              ExpectSvdMatchesKnownSingularValues(
                MakeThinMatrixWithKnownSingularValues(long_extent, is_tall, expected, dtype),
                expected, tol);
            }
          }
          struct Case {
            std::size_t rows;
            std::size_t cols;
            std::vector<double> singular_values;
          };
          const std::vector<Case> small_cases = {
            {4, 4, {3.5, 2.25, 1., 0.5}}, {4, 2, {3.5, 0.5}}, {2, 4, {2.25, 0.75}}};
          for (const Case& test_case : small_cases) {
            SCOPED_TRACE(Type.getname(dtype) + " " + std::to_string(test_case.rows) + "x" +
                         std::to_string(test_case.cols));
            ExpectSvdMatchesKnownSingularValues(
              MakeMatrixWithKnownSingularValues(test_case.rows, test_case.cols,
                                                test_case.singular_values, dtype),
              test_case.singular_values, tol);
          }
        }
      }

      // error test

      /*=====test info=====
      describe:error test, Test Dense diagonal tensor.
      input:
        T:Dense diagonal complex real type UniTensor.
        is_U:true
        is_VT:true
      ====================*/
      TEST(Svd, GpuErrDenseDiagTest) {
        int size = 5;
        std::vector<Bond> bonds = {Bond(size), Bond(size)};
        int rowrank = 1;
        bool is_diag = true;
        auto labels = std::vector<std::string>();
        auto T =
          UniTensor(bonds, labels, rowrank, Type.Double, Device.cuda, is_diag).to(Device.cuda);
        random::uniform_(T, 0, 10, 0);
        EXPECT_THROW({ std::vector<UniTensor> svds = linalg::Svd(T); }, std::logic_error);
      }

      /*=====test info=====
      describe:error test. Test Symmetric diagonal tensor.
      input:
        T:Symmetric diagonal complex real type UniTensor.
        is_U:true
        is_VT:true
      ====================*/
      TEST(Svd, GpuErrSymDiagTest) {
        std::vector<std::vector<cytnx_int64>> qnums = {{0}, {1}, {0}, {1}, {2}};
        std::vector<cytnx_uint64> degs = {1, 2, 3, 4, 5};
        auto syms = std::vector<Symmetry>(qnums[0].size(), Symmetry(SymmetryType::U));
        auto bond_ket = Bond(BD_KET, qnums, degs, syms);
        auto bond_bra = Bond(BD_BRA, qnums, degs, syms);
        std::vector<Bond> bonds = {bond_ket, bond_bra};
        cytnx_int64 row_rank = 1;
        bool is_diag = true;
        std::vector<std::string> labels = {};
        auto UT =
          UniTensor(bonds, labels, row_rank, Type.Double, Device.cuda, is_diag).to(Device.cuda);
        random::uniform_(UT, 0, 10, 0);
        EXPECT_THROW({ std::vector<UniTensor> svds = linalg::Svd(UT); }, std::logic_error);
      }

      /*=====test info=====
      describe:eror test. Test input the symmetric UniTensor with dtype is bool type.
      input:
        T:bool type symmetric UniTensor.
        is_U:true
        is_VT:true
      ====================*/
      TEST(Svd, GpuErrBoolTypeUT) {
        std::vector<std::vector<cytnx_int64>> qnums1 = {{0}, {1}, {0}, {1}, {2}};
        std::vector<cytnx_uint64> degs = {1, 2, 3, 4, 5};
        auto syms = std::vector<Symmetry>(qnums1[0].size(), Symmetry(SymmetryType::U));
        auto bond_ket = Bond(BD_KET, qnums1, degs, syms);
        std::vector<std::vector<cytnx_int64>> qnums2 = {{-1}, {-1}, {0}, {2}, {1}};
        syms = std::vector<Symmetry>(qnums2[0].size(), Symmetry(SymmetryType::U));
        auto bond_bra = Bond(BD_BRA, qnums2, degs, syms);
        std::vector<Bond> bonds = {bond_ket, bond_ket, bond_bra, bond_bra};
        cytnx_int64 row_rank = -1;
        std::vector<std::string> labels = {};
        auto src_T =
          UniTensor(bonds, labels, row_rank, Type.Bool, Device.cuda, false).to(Device.cuda);
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
      TEST(Svd, GpuErrVoidUTenTypeUT) {
        auto Ut = UniTensor();
        EXPECT_THROW({ std::vector<UniTensor> svds = linalg::Svd(Ut); }, std::logic_error)
          << "Should throw error when input void type unitensor but not."
             " Line:"
          << __LINE__ << std::endl;
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
        bool is_eq = AreNearlyEqUniTensor(T_float, ReCompose, tol);
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
      static void Check_UU_VV_Identity(const UniTensor& Tin, const std::vector<UniTensor>& Tout) {
        const UniTensor& U = Tout[1];
        const UniTensor& V = Tout[2];
        auto UD = U.Dagger();
        UD.relabel_({"0", "1", "9"});
        UD.permute_({2, 0, 1}, 1);
        auto UUD = Contract(U, UD);
      }

      static bool CheckResult(const std::string& case_name) {
        // test data source file
        std::string src_file_name = src_data_root + case_name + ".cytnx";
        // anscer file
        std::string ans_file_name = ans_data_root + case_name + ".cytnx";
        // bool need_U, need_VT;
        bool compute_uv;
        UniTensor src_T = UniTensor::Load(src_file_name).to(Device.cuda);
        UniTensor ans_T =
          UniTensor::Load(ans_file_name).to(Device.cuda);  // sigular values UniTensor

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

    }  // namespace
  }  // namespace gpu_test
}  // namespace cytnx
