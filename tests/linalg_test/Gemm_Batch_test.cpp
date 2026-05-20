#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "Device.hpp"
#include "Generator.hpp"
#include "Type.hpp"
#include "linalg.hpp"

namespace cytnx {
  namespace {

    // ── Reference computation ────────────────────────────────────────────────────

    // Row-major C = alpha*A*B + beta*C_in.  Returns result as a new vector.
    // A: m×k, B: k×n, C_in: m×n.
    std::vector<double> RefMatMul(int m, int n, int k, double alpha, double beta,
                                  const std::vector<double>& A, const std::vector<double>& B,
                                  const std::vector<double>& C_in) {
      std::vector<double> C = C_in;
      for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
          double acc = 0;
          for (int l = 0; l < k; l++) acc += A[i * k + l] * B[l * n + j];
          C[i * n + j] = alpha * acc + beta * C[i * n + j];
        }
      return C;
    }

    // ── Tensor helpers ───────────────────────────────────────────────────────────

    // Build a (rows×cols) Tensor of the given dtype from a flat row-major vector<double>.
    Tensor MakeTensor(int rows, int cols, unsigned int dtype, const std::vector<double>& data) {
      Tensor t = zeros({(cytnx_uint64)rows, (cytnx_uint64)cols}, dtype, Device.cpu);
      for (cytnx_uint64 i = 0; i < (cytnx_uint64)rows; i++)
        for (cytnx_uint64 j = 0; j < (cytnx_uint64)cols; j++)
          t.at({i, j}) = data[i * (cytnx_uint64)cols + j];
      return t;
    }

    // Run Gemm_Batch with a single group containing a single matrix.
    // Returns the result C tensor.
    Tensor RunSingle(const Scalar& alpha, const Scalar& beta, Tensor A, Tensor B, Tensor C) {
      std::vector<Tensor> as = {A}, bs = {B}, cs = {C};
      std::vector<Scalar> alphas = {alpha}, betas = {beta};
      linalg::Gemm_Batch(alphas, as, bs, betas, cs, {1});
      return cs[0];
    }

    void ExpectNearEl(double got, double expected, double rel, double abs_floor,
                      const std::string& label) {
      EXPECT_NEAR(got, expected, std::abs(expected) * rel + abs_floor) << label;
    }

  }  // namespace

  // ── Parameterized test types ──────────────────────────────────────────────────
  // Defined outside the anonymous namespace so INSTANTIATE_TEST_SUITE_P can see them.

  struct AlphaBetaCase {
    double alpha;
    double beta;
    std::string name;
  };

  struct DtypeCase {
    unsigned int dtype;
    double tol;
    std::string name;
  };

  // Tolerance per dtype for calculations with exact integer inputs.
  static const DtypeCase kAllDtypes[] = {
    {Type.Double, 1e-10, "Double"},
    {Type.Float, 1e-4, "Float"},
    {Type.ComplexDouble, 1e-10, "ComplexDouble"},
    {Type.ComplexFloat, 1e-4, "ComplexFloat"},
  };

#ifdef UNI_MKL

  // ═══════════════════════════════════════════════════════════════════════════════
  //  Parameterized: alpha / beta scalar boundary cases  (double matrices)
  // ═══════════════════════════════════════════════════════════════════════════════

  class GemmBatchScalarTest : public ::testing::TestWithParam<AlphaBetaCase> {};

  /*=====test info=====
  describe: Sweep over four critical alpha/beta combinations using the same 2×4 × 4×3 problem.
    alpha=1 beta=0  → C = A*B        (initial C must be fully discarded)
    alpha=0 beta=1  → C = C          (A, B have no effect; initial C preserved)
    alpha=0 beta=0  → C = 0          (both contributions nullified)
    alpha=2 beta=-1 → C = 2*A*B - C  (signed scaling of both terms)
  input: A 2×4, B 4×3 with structured data; C 2×3 initialized to 999.
  ====================*/
  TEST_P(GemmBatchScalarTest, DoubleMatrices) {
    const auto& p = GetParam();
    int m = 2, n = 3, k = 4;
    std::vector<double> ad = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<double> bd = {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};
    std::vector<double> cd(m * n, 999.0);

    Tensor C = RunSingle(Scalar(p.alpha), Scalar(p.beta), MakeTensor(m, k, Type.Double, ad),
                         MakeTensor(k, n, Type.Double, bd), MakeTensor(m, n, Type.Double, cd));
    auto ref = RefMatMul(m, n, k, p.alpha, p.beta, ad, bd, cd);

    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
        ExpectNearEl(C.at<double>({(cytnx_uint64)i, (cytnx_uint64)j}), ref[i * n + j], 1e-10, 1e-12,
                     "(" + std::to_string(i) + "," + std::to_string(j) + ")");
  }

  INSTANTIATE_TEST_SUITE_P(AlphaBetaCombinations, GemmBatchScalarTest,
                           ::testing::Values(AlphaBetaCase{1.0, 0.0, "Alpha1Beta0"},
                                             AlphaBetaCase{0.0, 1.0, "Alpha0Beta1"},
                                             AlphaBetaCase{0.0, 0.0, "Alpha0Beta0"},
                                             AlphaBetaCase{2.0, -1.0, "Alpha2BetaMinus1"}),
                           [](const ::testing::TestParamInfo<AlphaBetaCase>& info) {
                             return info.param.name;
                           });

  // ═══════════════════════════════════════════════════════════════════════════════
  //  Parameterized: data type coverage for A * I = A
  // ═══════════════════════════════════════════════════════════════════════════════

  class GemmBatchDtypeTest : public ::testing::TestWithParam<DtypeCase> {};

  /*=====test info=====
  describe: A * I = A for each of the four supported dtypes.
    Exercises the _d, _f, _cd, _cf dispatch paths with an analytically exact result.
  input: A=[[1,2],[3,4]], B=I2, alpha=1, beta=0. Expected: C == A.
  ====================*/
  TEST_P(GemmBatchDtypeTest, ATimesIdentity) {
    const auto& p = GetParam();
    int m = 2, n = 2, k = 2;
    Tensor A = zeros({(cytnx_uint64)m, (cytnx_uint64)k}, p.dtype, Device.cpu);
    Tensor B = zeros({(cytnx_uint64)k, (cytnx_uint64)n}, p.dtype, Device.cpu);
    Tensor C = zeros({(cytnx_uint64)m, (cytnx_uint64)n}, p.dtype, Device.cpu);
    A.at({(cytnx_uint64)0, (cytnx_uint64)0}) = 1;
    A.at({(cytnx_uint64)0, (cytnx_uint64)1}) = 2;
    A.at({(cytnx_uint64)1, (cytnx_uint64)0}) = 3;
    A.at({(cytnx_uint64)1, (cytnx_uint64)1}) = 4;
    B.at({(cytnx_uint64)0, (cytnx_uint64)0}) = 1;
    B.at({(cytnx_uint64)1, (cytnx_uint64)1}) = 1;

    std::vector<Tensor> as = {A}, bs = {B}, cs = {C};
    std::vector<Scalar> alphas = {Scalar(1.0)}, betas = {Scalar(0.0)};
    linalg::Gemm_Batch(alphas, as, bs, betas, cs, {1});

    Tensor res = cs[0].astype(Type.ComplexDouble);
    Tensor ref = A.astype(Type.ComplexDouble);
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++) {
        auto got = res.at<cytnx_complex128>({(cytnx_uint64)i, (cytnx_uint64)j});
        auto exp = ref.at<cytnx_complex128>({(cytnx_uint64)i, (cytnx_uint64)j});
        EXPECT_NEAR(got.real(), exp.real(), p.tol) << "(" << i << "," << j << ") real";
        EXPECT_NEAR(got.imag(), exp.imag(), p.tol) << "(" << i << "," << j << ") imag";
      }
  }

  INSTANTIATE_TEST_SUITE_P(AllDtypes, GemmBatchDtypeTest, ::testing::ValuesIn(kAllDtypes),
                           [](const ::testing::TestParamInfo<DtypeCase>& info) {
                             return info.param.name;
                           });

  // ═══════════════════════════════════════════════════════════════════════════════
  //  Matrix dimension boundary cases — parameterized over all dtypes
  // ═══════════════════════════════════════════════════════════════════════════════

  class GemmBatchDimTest : public ::testing::TestWithParam<DtypeCase> {};

  /*=====test info=====
  describe: 1×1×1 — degenerates to scalar arithmetic: C = alpha*a*b + beta*c.
  input: a=5, b=7, c=3, alpha=2, beta=4. Expected: 2*5*7 + 4*3 = 82.
  ====================*/
  TEST_P(GemmBatchDimTest, Dim1x1x1ScalarDegenerate) {
    const auto& p = GetParam();
    Tensor A = zeros({1, 1}, p.dtype, Device.cpu);
    A.at({(cytnx_uint64)0, (cytnx_uint64)0}) = 5.0;
    Tensor B = zeros({1, 1}, p.dtype, Device.cpu);
    B.at({(cytnx_uint64)0, (cytnx_uint64)0}) = 7.0;
    Tensor C_init = zeros({1, 1}, p.dtype, Device.cpu);
    C_init.at({(cytnx_uint64)0, (cytnx_uint64)0}) = 3.0;

    Tensor C = RunSingle(Scalar(2.0), Scalar(4.0), A, B, C_init);
    Tensor res = C.astype(Type.ComplexDouble);
    EXPECT_NEAR(res.at<cytnx_complex128>({(cytnx_uint64)0, (cytnx_uint64)0}).real(), 82.0, p.tol);
    EXPECT_NEAR(res.at<cytnx_complex128>({(cytnx_uint64)0, (cytnx_uint64)0}).imag(), 0.0, p.tol);
  }

  /*=====test info=====
  describe: k=1 (outer product). A is m×1, B is 1×n; the inner summation has exactly one term.
    C[i][j] = alpha * A[i][0] * B[0][j].
  input: A=[2,3,5]^T, B=[7,11,13,17], alpha=1, beta=0.
  ====================*/
  TEST_P(GemmBatchDimTest, DimK1OuterProduct) {
    const auto& p = GetParam();
    int m = 3, n = 4, k = 1;
    std::vector<double> ad = {2, 3, 5}, bd = {7, 11, 13, 17}, cd(m * n, 0.0);

    Tensor C = RunSingle(Scalar(1.0), Scalar(0.0), MakeTensor(m, k, p.dtype, ad),
                         MakeTensor(k, n, p.dtype, bd), MakeTensor(m, n, p.dtype, cd));
    auto ref = RefMatMul(m, n, k, 1.0, 0.0, ad, bd, cd);

    Tensor res = C.astype(Type.ComplexDouble);
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++) {
        auto got = res.at<cytnx_complex128>({(cytnx_uint64)i, (cytnx_uint64)j});
        ExpectNearEl(got.real(), ref[i * n + j], p.tol, p.tol,
                     "(" + std::to_string(i) + "," + std::to_string(j) + ")");
        EXPECT_NEAR(got.imag(), 0.0, p.tol);
      }
  }

  /*=====test info=====
  describe: B is the identity matrix → C = alpha*A + beta*C_0.
  input: A 3×3 with known data, B=I3, alpha=1.5, beta=0.5.
  ====================*/
  TEST_P(GemmBatchDimTest, DimBIsIdentity) {
    const auto& p = GetParam();
    int m = 3, n = 3, k = 3;
    std::vector<double> ad = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<double> bd = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    std::vector<double> cd = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    Tensor C = RunSingle(Scalar(1.5), Scalar(0.5), MakeTensor(m, k, p.dtype, ad),
                         MakeTensor(k, n, p.dtype, bd), MakeTensor(m, n, p.dtype, cd));
    auto ref = RefMatMul(m, n, k, 1.5, 0.5, ad, bd, cd);

    Tensor res = C.astype(Type.ComplexDouble);
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++) {
        auto got = res.at<cytnx_complex128>({(cytnx_uint64)i, (cytnx_uint64)j});
        ExpectNearEl(got.real(), ref[i * n + j], p.tol, p.tol,
                     "(" + std::to_string(i) + "," + std::to_string(j) + ")");
        EXPECT_NEAR(got.imag(), 0.0, p.tol);
      }
  }

  /*=====test info=====
  describe: Non-square matrices with m=3, n=5, k=2.
  ====================*/
  TEST_P(GemmBatchDimTest, DimNonSquare3x5K2) {
    const auto& p = GetParam();
    int m = 3, n = 5, k = 2;
    std::vector<double> ad(m * k), bd(k * n), cd(m * n, 0.0);
    for (int i = 0; i < m * k; i++) ad[i] = i + 1.0;
    for (int i = 0; i < k * n; i++) bd[i] = 1.0 / (i + 1.0);

    Tensor C = RunSingle(Scalar(1.0), Scalar(0.0), MakeTensor(m, k, p.dtype, ad),
                         MakeTensor(k, n, p.dtype, bd), MakeTensor(m, n, p.dtype, cd));
    auto ref = RefMatMul(m, n, k, 1.0, 0.0, ad, bd, cd);

    Tensor res = C.astype(Type.ComplexDouble);
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++) {
        auto got = res.at<cytnx_complex128>({(cytnx_uint64)i, (cytnx_uint64)j});
        ExpectNearEl(got.real(), ref[i * n + j], p.tol, p.tol,
                     "(" + std::to_string(i) + "," + std::to_string(j) + ")");
        EXPECT_NEAR(got.imag(), 0.0, p.tol);
      }
  }

  INSTANTIATE_TEST_SUITE_P(AllDtypes, GemmBatchDimTest, ::testing::ValuesIn(kAllDtypes),
                           [](const ::testing::TestParamInfo<DtypeCase>& info) {
                             return info.param.name;
                           });

  // ═══════════════════════════════════════════════════════════════════════════════
  //  Batch structure — parameterized over all dtypes
  // ═══════════════════════════════════════════════════════════════════════════════

  class GemmBatchBatchTest : public ::testing::TestWithParam<DtypeCase> {};

  /*=====test info=====
  describe: 3 groups of 1 matrix each with different per-group alpha and beta.
  input: groups 0/1/2 use (alpha=1,beta=0), (alpha=1,beta=1), (alpha=2,beta=0).
  ====================*/
  TEST_P(GemmBatchBatchTest, BatchOneGroupThreeMatrices) {
    const auto& p = GetParam();
    int m = 2, n = 2, k = 2;
    std::vector<double> a0 = {1, 0, 0, 1};
    std::vector<double> a1 = {1, 2, 3, 4};
    std::vector<double> a2 = {5, 6, 7, 8};
    std::vector<double> ball = {1, 1, 1, 1};
    std::vector<double> c0 = {0, 0, 0, 0};
    std::vector<double> c1 = {1, 2, 3, 4};
    std::vector<double> c2 = {0, 0, 0, 0};

    std::vector<Tensor> as = {MakeTensor(m, k, p.dtype, a0), MakeTensor(m, k, p.dtype, a1),
                              MakeTensor(m, k, p.dtype, a2)};
    std::vector<Tensor> bs = {MakeTensor(k, n, p.dtype, ball), MakeTensor(k, n, p.dtype, ball),
                              MakeTensor(k, n, p.dtype, ball)};
    std::vector<Tensor> cs = {MakeTensor(m, n, p.dtype, c0), MakeTensor(m, n, p.dtype, c1),
                              MakeTensor(m, n, p.dtype, c2)};
    std::vector<Scalar> alphas = {Scalar(1.0), Scalar(1.0), Scalar(2.0)};
    std::vector<Scalar> betas = {Scalar(0.0), Scalar(1.0), Scalar(0.0)};

    linalg::Gemm_Batch(alphas, as, bs, betas, cs, {1, 1, 1});

    const std::vector<double>* as_ref[] = {&a0, &a1, &a2};
    std::vector<double> cs_ref[] = {c0, c1, c2};
    double alphas_v[] = {1.0, 1.0, 2.0};
    double betas_v[] = {0.0, 1.0, 0.0};

    for (int idx = 0; idx < 3; idx++) {
      auto ref = RefMatMul(m, n, k, alphas_v[idx], betas_v[idx], *as_ref[idx], ball, cs_ref[idx]);
      Tensor res = cs[idx].astype(Type.ComplexDouble);
      for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
          auto got = res.at<cytnx_complex128>({(cytnx_uint64)i, (cytnx_uint64)j});
          ExpectNearEl(got.real(), ref[i * n + j], p.tol, p.tol,
                       "matrix=" + std::to_string(idx) + " (" + std::to_string(i) + "," +
                         std::to_string(j) + ")");
          EXPECT_NEAR(got.imag(), 0.0, p.tol);
        }
    }
  }

  /*=====test info=====
  describe: 2 groups with different matrix dimensions.
    Group 0: 2 matrices of size 2×3 (k=2).
    Group 1: 1 matrix of size 3×3 (k=3), set to I×I = I.
  ====================*/
  TEST_P(GemmBatchBatchTest, BatchTwoGroupsDifferentDims) {
    const auto& p = GetParam();
    int m0 = 2, n0 = 3, k0 = 2;
    std::vector<double> a0 = {1, 2, 3, 4};
    std::vector<double> b0 = {1, 0, 1, 0, 1, 0};
    std::vector<double> c0(m0 * n0, 0.0);
    auto ref_g0 = RefMatMul(m0, n0, k0, 1.0, 0.0, a0, b0, c0);

    int m1 = 3, n1 = 3, k1 = 3;
    std::vector<double> eye3 = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    std::vector<double> c1(m1 * n1, 0.0);

    std::vector<Tensor> as = {MakeTensor(m0, k0, p.dtype, a0), MakeTensor(m0, k0, p.dtype, a0),
                              MakeTensor(m1, k1, p.dtype, eye3)};
    std::vector<Tensor> bs = {MakeTensor(k0, n0, p.dtype, b0), MakeTensor(k0, n0, p.dtype, b0),
                              MakeTensor(k1, n1, p.dtype, eye3)};
    std::vector<Tensor> cs = {MakeTensor(m0, n0, p.dtype, c0), MakeTensor(m0, n0, p.dtype, c0),
                              MakeTensor(m1, n1, p.dtype, c1)};
    std::vector<Scalar> alphas = {Scalar(1.0), Scalar(1.0)};
    std::vector<Scalar> betas = {Scalar(0.0), Scalar(0.0)};

    linalg::Gemm_Batch(alphas, as, bs, betas, cs, {2, 1});

    for (int idx = 0; idx < 2; idx++) {
      Tensor res = cs[idx].astype(Type.ComplexDouble);
      for (int i = 0; i < m0; i++)
        for (int j = 0; j < n0; j++) {
          auto got = res.at<cytnx_complex128>({(cytnx_uint64)i, (cytnx_uint64)j});
          ExpectNearEl(got.real(), ref_g0[i * n0 + j], p.tol, p.tol,
                       "group0 matrix=" + std::to_string(idx) + " (" + std::to_string(i) + "," +
                         std::to_string(j) + ")");
          EXPECT_NEAR(got.imag(), 0.0, p.tol);
        }
    }
    // I*I = I
    Tensor res2 = cs[2].astype(Type.ComplexDouble);
    for (int i = 0; i < m1; i++)
      for (int j = 0; j < n1; j++) {
        auto got = res2.at<cytnx_complex128>({(cytnx_uint64)i, (cytnx_uint64)j});
        EXPECT_NEAR(got.real(), (i == j ? 1.0 : 0.0), p.tol) << "group1 (" << i << "," << j << ")";
        EXPECT_NEAR(got.imag(), 0.0, p.tol);
      }
  }

  INSTANTIATE_TEST_SUITE_P(AllDtypes, GemmBatchBatchTest, ::testing::ValuesIn(kAllDtypes),
                           [](const ::testing::TestParamInfo<DtypeCase>& info) {
                             return info.param.name;
                           });

  // ═══════════════════════════════════════════════════════════════════════════════
  //  Complex128 imaginary arithmetic (single dtype, not parameterized)
  // ═══════════════════════════════════════════════════════════════════════════════

  /*=====test info=====
  describe: Complex128 — imaginary parts propagate correctly.
    A = [[i, 0],[0, 1]], B = [[1, i],[0, 1]], alpha=1, beta=0.
    C[0][0]=i, C[0][1]=-1, C[1][0]=0, C[1][1]=1.
  ====================*/
  TEST(GemmBatch, DtypeComplex128ImaginaryArithmetic) {
    using C128 = cytnx_complex128;
    int m = 2, n = 2, k = 2;
    Tensor A = zeros({(cytnx_uint64)m, (cytnx_uint64)k}, Type.ComplexDouble, Device.cpu);
    Tensor B = zeros({(cytnx_uint64)k, (cytnx_uint64)n}, Type.ComplexDouble, Device.cpu);
    Tensor C = zeros({(cytnx_uint64)m, (cytnx_uint64)n}, Type.ComplexDouble, Device.cpu);
    A.at({(cytnx_uint64)0, (cytnx_uint64)0}) = C128(0, 1);
    A.at({(cytnx_uint64)1, (cytnx_uint64)1}) = C128(1, 0);
    B.at({(cytnx_uint64)0, (cytnx_uint64)0}) = C128(1, 0);
    B.at({(cytnx_uint64)0, (cytnx_uint64)1}) = C128(0, 1);
    B.at({(cytnx_uint64)1, (cytnx_uint64)1}) = C128(1, 0);

    std::vector<Tensor> as = {A}, bs = {B}, cs = {C};
    std::vector<Scalar> alphas = {Scalar(C128(1, 0))}, betas = {Scalar(C128(0, 0))};
    linalg::Gemm_Batch(alphas, as, bs, betas, cs, {1});

    EXPECT_NEAR(cs[0].at<C128>({(cytnx_uint64)0, (cytnx_uint64)0}).real(), 0.0, 1e-12);
    EXPECT_NEAR(cs[0].at<C128>({(cytnx_uint64)0, (cytnx_uint64)0}).imag(), 1.0, 1e-12);
    EXPECT_NEAR(cs[0].at<C128>({(cytnx_uint64)0, (cytnx_uint64)1}).real(), -1.0, 1e-12);
    EXPECT_NEAR(cs[0].at<C128>({(cytnx_uint64)0, (cytnx_uint64)1}).imag(), 0.0, 1e-12);
    EXPECT_NEAR(cs[0].at<C128>({(cytnx_uint64)1, (cytnx_uint64)0}).real(), 0.0, 1e-12);
    EXPECT_NEAR(cs[0].at<C128>({(cytnx_uint64)1, (cytnx_uint64)0}).imag(), 0.0, 1e-12);
    EXPECT_NEAR(cs[0].at<C128>({(cytnx_uint64)1, (cytnx_uint64)1}).real(), 1.0, 1e-12);
    EXPECT_NEAR(cs[0].at<C128>({(cytnx_uint64)1, (cytnx_uint64)1}).imag(), 0.0, 1e-12);
  }

  // ═══════════════════════════════════════════════════════════════════════════════
  //  Type promotion — parameterized
  // ═══════════════════════════════════════════════════════════════════════════════

  struct TypePromotionCase {
    unsigned int tensor_dtype_a;
    unsigned int tensor_dtype_b;
    unsigned int scalar_dtype;  // dtype of the alpha/beta Scalar
    unsigned int expected_dtype;
    double tol;
    std::string name;
  };

  class GemmBatchTypePromotionTest : public ::testing::TestWithParam<TypePromotionCase> {};

  /*=====test info=====
  describe: A * I = A for various tensor/scalar dtype combinations.
    Verifies that Gemm_Batch promotes operands to the highest dtype present and
    that the result dtype matches expected_dtype.
  ====================*/
  TEST_P(GemmBatchTypePromotionTest, ATimesIdentityPromotion) {
    const auto& p = GetParam();
    int m = 2, n = 2, k = 2;
    Tensor A = zeros({(cytnx_uint64)m, (cytnx_uint64)k}, p.tensor_dtype_a, Device.cpu);
    Tensor B = zeros({(cytnx_uint64)k, (cytnx_uint64)n}, p.tensor_dtype_b, Device.cpu);
    Tensor C = zeros({(cytnx_uint64)m, (cytnx_uint64)n}, p.tensor_dtype_a, Device.cpu);
    A.at({(cytnx_uint64)0, (cytnx_uint64)0}) = 1;
    A.at({(cytnx_uint64)0, (cytnx_uint64)1}) = 2;
    A.at({(cytnx_uint64)1, (cytnx_uint64)0}) = 3;
    A.at({(cytnx_uint64)1, (cytnx_uint64)1}) = 4;
    B.at({(cytnx_uint64)0, (cytnx_uint64)0}) = 1;
    B.at({(cytnx_uint64)1, (cytnx_uint64)1}) = 1;

    // alpha/beta are scalars of the specified dtype
    Scalar alpha, beta;
    if (p.scalar_dtype == Type.ComplexDouble) {
      alpha = Scalar(cytnx_complex128(1, 0));
      beta = Scalar(cytnx_complex128(0, 0));
    } else if (p.scalar_dtype == Type.ComplexFloat) {
      alpha = Scalar(cytnx_complex64(1, 0));
      beta = Scalar(cytnx_complex64(0, 0));
    } else if (p.scalar_dtype == Type.Double) {
      alpha = Scalar(1.0);
      beta = Scalar(0.0);
    } else {
      alpha = Scalar(1.0f);
      beta = Scalar(0.0f);
    }

    std::vector<Tensor> as = {A}, bs = {B}, cs = {C};
    std::vector<Scalar> alphas = {alpha}, betas = {beta};
    linalg::Gemm_Batch(alphas, as, bs, betas, cs, {1});

    EXPECT_EQ(cs[0].dtype(), (unsigned int)p.expected_dtype)
      << "expected dtype " << p.expected_dtype << " but got " << cs[0].dtype();

    Tensor res = cs[0].astype(Type.ComplexDouble);
    double exp_vals[2][2] = {{1, 2}, {3, 4}};
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++) {
        auto got = res.at<cytnx_complex128>({(cytnx_uint64)i, (cytnx_uint64)j});
        EXPECT_NEAR(got.real(), exp_vals[i][j], p.tol) << "(" << i << "," << j << ") real";
        EXPECT_NEAR(got.imag(), 0.0, p.tol) << "(" << i << "," << j << ") imag";
      }
  }

  INSTANTIATE_TEST_SUITE_P(
    MixedTypes, GemmBatchTypePromotionTest,
    ::testing::Values(
      // float tensors + double scalars → double
      TypePromotionCase{Type.Float, Type.Float, Type.Double, Type.Double, 1e-10,
                        "FloatTensorsDoubleScalar"},
      // double tensors + complex128 scalar → complex128
      TypePromotionCase{Type.Double, Type.Double, Type.ComplexDouble, Type.ComplexDouble, 1e-10,
                        "DoubleTensorsComplex128Scalar"},
      // float tensors + complex128 scalar → complex128
      TypePromotionCase{Type.Float, Type.Float, Type.ComplexDouble, Type.ComplexDouble, 1e-4,
                        "FloatTensorsComplex128Scalar"},
      // float tensors + complex64 scalar → complex64
      TypePromotionCase{Type.Float, Type.Float, Type.ComplexFloat, Type.ComplexFloat, 1e-4,
                        "FloatTensorsComplex64Scalar"},
      // mixed tensor dtypes: float A + double B → double
      TypePromotionCase{Type.Float, Type.Double, Type.Double, Type.Double, 1e-10,
                        "FloatATensorDoubleB"}),
    [](const ::testing::TestParamInfo<TypePromotionCase>& info) { return info.param.name; });

#else  // !UNI_MKL

  // ═══════════════════════════════════════════════════════════════════════════════
  //  Without MKL: all four dispatch paths must throw std::logic_error
  // ═══════════════════════════════════════════════════════════════════════════════

  class GemmBatchNoMKLTest : public ::testing::TestWithParam<DtypeCase> {};

  /*=====test info=====
  describe: Gemm_Batch_internal calls cytnx_error_msg(true,...) when MKL is absent,
    which throws std::logic_error.  One case per supported dtype.
  ====================*/
  TEST_P(GemmBatchNoMKLTest, ThrowsWhenMklAbsent) {
    const auto& p = GetParam();
    Tensor A = zeros({2, 2}, p.dtype, Device.cpu);
    Tensor B = zeros({2, 2}, p.dtype, Device.cpu);
    Tensor C = zeros({2, 2}, p.dtype, Device.cpu);
    std::vector<Tensor> as = {A}, bs = {B}, cs = {C};
    // Scalar is double; promotion will pick the tensor dtype.
    std::vector<Scalar> alphas = {Scalar(1.0)}, betas = {Scalar(0.0)};
    EXPECT_THROW(linalg::Gemm_Batch(alphas, as, bs, betas, cs, {1}), std::logic_error);
  }

  INSTANTIATE_TEST_SUITE_P(AllDtypes, GemmBatchNoMKLTest, ::testing::ValuesIn(kAllDtypes),
                           [](const ::testing::TestParamInfo<DtypeCase>& info) {
                             return info.param.name;
                           });

#endif  // UNI_MKL

  // ═══════════════════════════════════════════════════════════════════════════════
  //  Error cases — dimension and group-structure inconsistency
  //
  //  All structural checks are unconditional (no User_debug gate).  They throw
  //  std::logic_error before any BLAS call, so these tests work with or without MKL.
  // ═══════════════════════════════════════════════════════════════════════════════

  /*=====test info=====
  describe: A.cols (k) != B.rows violates the fundamental GEMM contraction dimension.
    A is 2×4, B is 5×3: B has 5 rows but A has only 4 columns.
  ====================*/
  TEST(GemmBatchError, ThrowsOnInnerDimMismatch) {
    Tensor A = zeros({2, 4}, Type.Double, Device.cpu);
    Tensor B = zeros({5, 3}, Type.Double, Device.cpu);  // wrong: rows should be 4
    Tensor C = zeros({2, 3}, Type.Double, Device.cpu);
    std::vector<Tensor> as = {A}, bs = {B}, cs = {C};
    std::vector<Scalar> alphas = {Scalar(1.0)}, betas = {Scalar(0.0)};
    EXPECT_THROW(linalg::Gemm_Batch(alphas, as, bs, betas, cs, {1}), std::logic_error);
  }

  /*=====test info=====
  describe: A.rows != C.rows: the output tensor has the wrong number of rows.
    A is 2×3, C is 4×4: C has 4 rows but A has only 2.
  ====================*/
  TEST(GemmBatchError, ThrowsOnOutputRowMismatch) {
    Tensor A = zeros({2, 3}, Type.Double, Device.cpu);
    Tensor B = zeros({3, 4}, Type.Double, Device.cpu);
    Tensor C = zeros({4, 4}, Type.Double, Device.cpu);  // wrong: rows should be 2
    std::vector<Tensor> as = {A}, bs = {B}, cs = {C};
    std::vector<Scalar> alphas = {Scalar(1.0)}, betas = {Scalar(0.0)};
    EXPECT_THROW(linalg::Gemm_Batch(alphas, as, bs, betas, cs, {1}), std::logic_error);
  }

  /*=====test info=====
  describe: B.cols != C.cols: the output tensor has the wrong number of columns.
    B is 3×4, C is 2×7: C has 7 columns but B has only 4.
  ====================*/
  TEST(GemmBatchError, ThrowsOnOutputColMismatch) {
    Tensor A = zeros({2, 3}, Type.Double, Device.cpu);
    Tensor B = zeros({3, 4}, Type.Double, Device.cpu);
    Tensor C = zeros({2, 7}, Type.Double, Device.cpu);  // wrong: cols should be 4
    std::vector<Tensor> as = {A}, bs = {B}, cs = {C};
    std::vector<Scalar> alphas = {Scalar(1.0)}, betas = {Scalar(0.0)};
    EXPECT_THROW(linalg::Gemm_Batch(alphas, as, bs, betas, cs, {1}), std::logic_error);
  }

  /*=====test info=====
  describe: Number of tensors in as/bs/cs does not equal the sum of group_sizes.
    group_size={3} implies 3 matrices, but only 1 tensor is provided.
  ====================*/
  TEST(GemmBatchError, ThrowsOnTensorCountMismatch) {
    Tensor A = zeros({2, 2}, Type.Double, Device.cpu);
    Tensor B = zeros({2, 2}, Type.Double, Device.cpu);
    Tensor C = zeros({2, 2}, Type.Double, Device.cpu);
    std::vector<Tensor> as = {A}, bs = {B}, cs = {C};
    std::vector<Scalar> alphas = {Scalar(1.0)}, betas = {Scalar(0.0)};
    // group_size={3} → 3 matrices expected, but we only provided 1
    EXPECT_THROW(linalg::Gemm_Batch(alphas, as, bs, betas, cs, {3}), std::logic_error);
  }

  /*=====test info=====
  describe: Two matrices in the same group have different m dimensions (rows of A).
    Group has 2 matrices: first is 2×2, second is 3×2 — rows differ.
  ====================*/
  TEST(GemmBatchError, ThrowsOnWithinGroupDimMismatch) {
    Tensor A0 = zeros({2, 2}, Type.Double, Device.cpu);
    Tensor A1 = zeros({3, 2}, Type.Double, Device.cpu);  // wrong: rows should be 2
    Tensor B0 = zeros({2, 2}, Type.Double, Device.cpu);
    Tensor B1 = zeros({2, 2}, Type.Double, Device.cpu);
    Tensor C0 = zeros({2, 2}, Type.Double, Device.cpu);
    Tensor C1 = zeros({3, 2}, Type.Double, Device.cpu);
    std::vector<Tensor> as = {A0, A1}, bs = {B0, B1}, cs = {C0, C1};
    std::vector<Scalar> alphas = {Scalar(1.0)};  // one per group
    std::vector<Scalar> betas = {Scalar(0.0)};  // one per group
    EXPECT_THROW(linalg::Gemm_Batch(alphas, as, bs, betas, cs, {2}), std::logic_error);
  }

  // ── Invalid scalar type tests ─────────────────────────────────────────────────

  /*=====test info=====
  describe: alpha with an integer dtype (Int64, dtype=5 > 4) is rejected.
    Only (complex/real)(double/float) scalars are accepted.
  ====================*/
  TEST(GemmBatchError, ThrowsOnIntegerAlpha) {
    Tensor A = zeros({2, 2}, Type.Double, Device.cpu);
    Tensor B = zeros({2, 2}, Type.Double, Device.cpu);
    Tensor C = zeros({2, 2}, Type.Double, Device.cpu);
    std::vector<Tensor> as = {A}, bs = {B}, cs = {C};
    std::vector<Scalar> alphas = {Scalar(cytnx_int64(1))};  // integer type: dtype=5 > 4
    std::vector<Scalar> betas = {Scalar(0.0)};
    EXPECT_THROW(linalg::Gemm_Batch(alphas, as, bs, betas, cs, {1}), std::logic_error);
  }

  /*=====test info=====
  describe: beta with a Void dtype (default-constructed Scalar) is rejected.
  ====================*/
  TEST(GemmBatchError, ThrowsOnVoidBeta) {
    Tensor A = zeros({2, 2}, Type.Double, Device.cpu);
    Tensor B = zeros({2, 2}, Type.Double, Device.cpu);
    Tensor C = zeros({2, 2}, Type.Double, Device.cpu);
    std::vector<Tensor> as = {A}, bs = {B}, cs = {C};
    std::vector<Scalar> alphas = {Scalar(1.0)};
    std::vector<Scalar> betas = {Scalar()};  // default-constructed → Void dtype
    EXPECT_THROW(linalg::Gemm_Batch(alphas, as, bs, betas, cs, {1}), std::logic_error);
  }

}  // namespace cytnx
