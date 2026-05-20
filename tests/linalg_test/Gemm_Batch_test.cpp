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
    Tensor RunSingle(int m, int n, int k, const Scalar& alpha, const Scalar& beta, Tensor A,
                     Tensor B, Tensor C) {
      std::vector<Tensor> as = {A}, bs = {B}, cs = {C};
      std::vector<Scalar> alphas = {alpha}, betas = {beta};
      linalg::Gemm_Batch({(cytnx_int64)m}, {(cytnx_int64)n}, {(cytnx_int64)k}, alphas, as, bs,
                         betas, cs, 1, {1});
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

  // ── Debug-mode fixture ────────────────────────────────────────────────────────
  // Dimension-mismatch and group-count checks in linalg::Gemm_Batch live inside
  // `if (User_debug)` (Gemm_Batch.cpp:104).  This fixture enables that guard for
  // the duration of each test, then restores it.  The actual validation throws
  // before any MKL call, so these tests compile and run with or without MKL.

  class GemmBatchDebugTest : public ::testing::Test {
   protected:
    void SetUp() override { User_debug = true; }
    void TearDown() override { User_debug = false; }
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

    Tensor C =
      RunSingle(m, n, k, Scalar(p.alpha), Scalar(p.beta), MakeTensor(m, k, Type.Double, ad),
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
  //  Matrix dimension boundary cases
  // ═══════════════════════════════════════════════════════════════════════════════

  /*=====test info=====
  describe: 1×1×1 — degenerates to scalar arithmetic: C = alpha*a*b + beta*c.
  input: a=5, b=7, c=3, alpha=2, beta=4. Expected: 2*5*7 + 4*3 = 82.
  ====================*/
  TEST(GemmBatch, Dim1x1x1ScalarDegenerate) {
    Tensor A = zeros({1, 1}, Type.Double, Device.cpu);
    A.at({(cytnx_uint64)0, (cytnx_uint64)0}) = 5.0;
    Tensor B = zeros({1, 1}, Type.Double, Device.cpu);
    B.at({(cytnx_uint64)0, (cytnx_uint64)0}) = 7.0;
    Tensor C_init = zeros({1, 1}, Type.Double, Device.cpu);
    C_init.at({(cytnx_uint64)0, (cytnx_uint64)0}) = 3.0;

    Tensor C = RunSingle(1, 1, 1, Scalar(2.0), Scalar(4.0), A, B, C_init);
    EXPECT_NEAR(C.at<double>({(cytnx_uint64)0, (cytnx_uint64)0}), 82.0, 1e-10);
  }

  /*=====test info=====
  describe: k=1 (outer product). A is m×1, B is 1×n; the inner summation has exactly one term.
    C[i][j] = alpha * A[i][0] * B[0][j].
  input: A=[2,3,5]^T, B=[7,11,13,17], alpha=1, beta=0.
  ====================*/
  TEST(GemmBatch, DimK1OuterProduct) {
    int m = 3, n = 4, k = 1;
    std::vector<double> ad = {2, 3, 5}, bd = {7, 11, 13, 17}, cd(m * n, 0.0);

    Tensor C = RunSingle(m, n, k, Scalar(1.0), Scalar(0.0), MakeTensor(m, k, Type.Double, ad),
                         MakeTensor(k, n, Type.Double, bd), MakeTensor(m, n, Type.Double, cd));
    auto ref = RefMatMul(m, n, k, 1.0, 0.0, ad, bd, cd);

    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
        ExpectNearEl(C.at<double>({(cytnx_uint64)i, (cytnx_uint64)j}), ref[i * n + j], 1e-10, 1e-12,
                     "(" + std::to_string(i) + "," + std::to_string(j) + ")");
  }

  /*=====test info=====
  describe: B is the identity matrix → C = alpha*A + beta*C_0.
    Multiplying by the identity is a useful boundary that checks no extra additions occur.
  input: A 3×3 with known data, B=I3, alpha=1.5, beta=0.5.
  ====================*/
  TEST(GemmBatch, DimBIsIdentity) {
    int m = 3, n = 3, k = 3;
    std::vector<double> ad = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<double> bd = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // identity
    std::vector<double> cd = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

    Tensor C = RunSingle(m, n, k, Scalar(1.5), Scalar(0.5), MakeTensor(m, k, Type.Double, ad),
                         MakeTensor(k, n, Type.Double, bd), MakeTensor(m, n, Type.Double, cd));
    auto ref = RefMatMul(m, n, k, 1.5, 0.5, ad, bd, cd);

    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
        ExpectNearEl(C.at<double>({(cytnx_uint64)i, (cytnx_uint64)j}), ref[i * n + j], 1e-10, 1e-12,
                     "(" + std::to_string(i) + "," + std::to_string(j) + ")");
  }

  /*=====test info=====
  describe: Non-square matrices with m=3, n=5, k=2. Tests rectangular allocation and indexing.
  ====================*/
  TEST(GemmBatch, DimNonSquare3x5K2) {
    int m = 3, n = 5, k = 2;
    std::vector<double> ad(m * k), bd(k * n), cd(m * n, 0.0);
    for (int i = 0; i < m * k; i++) ad[i] = i + 1.0;
    for (int i = 0; i < k * n; i++) bd[i] = 1.0 / (i + 1.0);

    Tensor C = RunSingle(m, n, k, Scalar(1.0), Scalar(0.0), MakeTensor(m, k, Type.Double, ad),
                         MakeTensor(k, n, Type.Double, bd), MakeTensor(m, n, Type.Double, cd));
    auto ref = RefMatMul(m, n, k, 1.0, 0.0, ad, bd, cd);

    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
        ExpectNearEl(C.at<double>({(cytnx_uint64)i, (cytnx_uint64)j}), ref[i * n + j], 1e-9, 1e-12,
                     "(" + std::to_string(i) + "," + std::to_string(j) + ")");
  }

  // ═══════════════════════════════════════════════════════════════════════════════
  //  Batch structure
  // ═══════════════════════════════════════════════════════════════════════════════

  /*=====test info=====
  describe: 1 group, 3 matrices of equal size but with different per-matrix alpha and beta.
    Verifies that each matrix in the batch is processed independently with its own scalars.
  input: matrices 0/1/2 use (alpha=1,beta=0), (alpha=1,beta=1), (alpha=2,beta=0) respectively.
  ====================*/
  TEST(GemmBatch, BatchOneGroupThreeMatrices) {
    int m = 2, n = 2, k = 2;
    std::vector<double> a0 = {1, 0, 0, 1};  // identity
    std::vector<double> a1 = {1, 2, 3, 4};
    std::vector<double> a2 = {5, 6, 7, 8};
    std::vector<double> ball = {1, 1, 1, 1};  // all-ones, shared
    std::vector<double> c0 = {0, 0, 0, 0};
    std::vector<double> c1 = {1, 2, 3, 4};
    std::vector<double> c2 = {0, 0, 0, 0};

    std::vector<Tensor> as = {MakeTensor(m, k, Type.Double, a0), MakeTensor(m, k, Type.Double, a1),
                              MakeTensor(m, k, Type.Double, a2)};
    std::vector<Tensor> bs = {MakeTensor(k, n, Type.Double, ball),
                              MakeTensor(k, n, Type.Double, ball),
                              MakeTensor(k, n, Type.Double, ball)};
    std::vector<Tensor> cs = {MakeTensor(m, n, Type.Double, c0), MakeTensor(m, n, Type.Double, c1),
                              MakeTensor(m, n, Type.Double, c2)};
    std::vector<Scalar> alphas = {Scalar(1.0), Scalar(1.0), Scalar(2.0)};
    std::vector<Scalar> betas = {Scalar(0.0), Scalar(1.0), Scalar(0.0)};
    std::vector<cytnx_int64> ms(3, m), ns(3, n), ks(3, k);

    linalg::Gemm_Batch(ms, ns, ks, alphas, as, bs, betas, cs, 3, {1, 1, 1});

    const std::vector<double>* as_ref[] = {&a0, &a1, &a2};
    std::vector<double> cs_ref[] = {c0, c1, c2};
    double alphas_v[] = {1.0, 1.0, 2.0};
    double betas_v[] = {0.0, 1.0, 0.0};

    for (int idx = 0; idx < 3; idx++) {
      auto ref = RefMatMul(m, n, k, alphas_v[idx], betas_v[idx], *as_ref[idx], ball, cs_ref[idx]);
      for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
          ExpectNearEl(cs[idx].at<double>({(cytnx_uint64)i, (cytnx_uint64)j}), ref[i * n + j],
                       1e-10, 1e-12,
                       "matrix=" + std::to_string(idx) + " (" + std::to_string(i) + "," +
                         std::to_string(j) + ")");
    }
  }

  /*=====test info=====
  describe: 2 groups with different matrix dimensions.
    Group 0: 2 matrices of size 2×3 (k=2).
    Group 1: 1 matrix of size 3×3 (k=3), set to I×I = I to give an analytically known result.
  ====================*/
  TEST(GemmBatch, BatchTwoGroupsDifferentDims) {
    int m0 = 2, n0 = 3, k0 = 2;
    std::vector<double> a0 = {1, 2, 3, 4};
    std::vector<double> b0 = {1, 0, 1, 0, 1, 0};
    std::vector<double> c0(m0 * n0, 0.0);
    auto ref_g0 = RefMatMul(m0, n0, k0, 1.0, 0.0, a0, b0, c0);

    int m1 = 3, n1 = 3, k1 = 3;
    std::vector<double> eye3 = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    std::vector<double> c1(m1 * n1, 0.0);

    std::vector<Tensor> as = {MakeTensor(m0, k0, Type.Double, a0),
                              MakeTensor(m0, k0, Type.Double, a0),
                              MakeTensor(m1, k1, Type.Double, eye3)};
    std::vector<Tensor> bs = {MakeTensor(k0, n0, Type.Double, b0),
                              MakeTensor(k0, n0, Type.Double, b0),
                              MakeTensor(k1, n1, Type.Double, eye3)};
    std::vector<Tensor> cs = {MakeTensor(m0, n0, Type.Double, c0),
                              MakeTensor(m0, n0, Type.Double, c0),
                              MakeTensor(m1, n1, Type.Double, c1)};
    std::vector<Scalar> alphas = {Scalar(1.0), Scalar(1.0), Scalar(1.0)};
    std::vector<Scalar> betas = {Scalar(0.0), Scalar(0.0), Scalar(0.0)};
    std::vector<cytnx_int64> ms = {m0, m1}, ns = {n0, n1}, ks = {k0, k1};

    linalg::Gemm_Batch(ms, ns, ks, alphas, as, bs, betas, cs, 2, {2, 1});

    for (int idx = 0; idx < 2; idx++)
      for (int i = 0; i < m0; i++)
        for (int j = 0; j < n0; j++)
          ExpectNearEl(cs[idx].at<double>({(cytnx_uint64)i, (cytnx_uint64)j}), ref_g0[i * n0 + j],
                       1e-10, 1e-12,
                       "group0 matrix=" + std::to_string(idx) + " (" + std::to_string(i) + "," +
                         std::to_string(j) + ")");

    // I*I = I
    for (int i = 0; i < m1; i++)
      for (int j = 0; j < n1; j++)
        EXPECT_NEAR(cs[2].at<double>({(cytnx_uint64)i, (cytnx_uint64)j}), (i == j ? 1.0 : 0.0),
                    1e-12)
          << "group1 (" << i << "," << j << ")";
  }

  // ═══════════════════════════════════════════════════════════════════════════════
  //  Parameterized: data type coverage
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
    linalg::Gemm_Batch({(cytnx_int64)m}, {(cytnx_int64)n}, {(cytnx_int64)k}, alphas, as, bs, betas,
                       cs, 1, {1});

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

  INSTANTIATE_TEST_SUITE_P(AllDtypes, GemmBatchDtypeTest,
                           ::testing::Values(DtypeCase{Type.Double, 1e-10, "Double"},
                                             DtypeCase{Type.Float, 1e-4, "Float"},
                                             DtypeCase{Type.ComplexDouble, 1e-10, "ComplexDouble"},
                                             DtypeCase{Type.ComplexFloat, 1e-4, "ComplexFloat"}),
                           [](const ::testing::TestParamInfo<DtypeCase>& info) {
                             return info.param.name;
                           });

  /*=====test info=====
  describe: Complex128 — imaginary parts propagate correctly through the multiply-add.
    A = [[i, 0],[0, 1]], B = [[1, i],[0, 1]], alpha=1, beta=0.
    C[0][0] = i*1 = i  (real=0, imag=1)
    C[0][1] = i*i = -1 (real=-1, imag=0)
    C[1][0] = 0
    C[1][1] = 1*1 = 1  (real=1, imag=0)
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
    linalg::Gemm_Batch({(cytnx_int64)m}, {(cytnx_int64)n}, {(cytnx_int64)k}, alphas, as, bs, betas,
                       cs, 1, {1});

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
  //  Type promotion
  // ═══════════════════════════════════════════════════════════════════════════════

  /*=====test info=====
  describe: Float tensors + double alpha/beta → result promoted to double (fin_dtype=Double).
    Verifies that Gemm_Batch upcasts all operands to the highest-precision type present.
  input: A=[[1,2],[3,4]] (float), B=I2 (float), alpha=1.0 (double), beta=0.0 (double).
  Expected: result dtype == Double, values equal A.
  ====================*/
  TEST(GemmBatch, TypePromotionFloatTensorsDoubleScalars) {
    int m = 2, n = 2, k = 2;
    Tensor A = zeros({(cytnx_uint64)m, (cytnx_uint64)k}, Type.Float, Device.cpu);
    Tensor B = zeros({(cytnx_uint64)k, (cytnx_uint64)n}, Type.Float, Device.cpu);
    Tensor C = zeros({(cytnx_uint64)m, (cytnx_uint64)n}, Type.Float, Device.cpu);
    A.at({(cytnx_uint64)0, (cytnx_uint64)0}) = 1.0f;
    A.at({(cytnx_uint64)0, (cytnx_uint64)1}) = 2.0f;
    A.at({(cytnx_uint64)1, (cytnx_uint64)0}) = 3.0f;
    A.at({(cytnx_uint64)1, (cytnx_uint64)1}) = 4.0f;
    B.at({(cytnx_uint64)0, (cytnx_uint64)0}) = 1.0f;
    B.at({(cytnx_uint64)1, (cytnx_uint64)1}) = 1.0f;

    std::vector<Tensor> as = {A}, bs = {B}, cs = {C};
    std::vector<Scalar> alphas = {Scalar(1.0)}, betas = {Scalar(0.0)};  // double scalars
    linalg::Gemm_Batch({(cytnx_int64)m}, {(cytnx_int64)n}, {(cytnx_int64)k}, alphas, as, bs, betas,
                       cs, 1, {1});

    EXPECT_EQ(cs[0].dtype(), (unsigned int)Type.Double)
      << "float tensors with double scalars must produce a double result";
    EXPECT_NEAR(cs[0].at<double>({(cytnx_uint64)0, (cytnx_uint64)0}), 1.0, 1e-10);
    EXPECT_NEAR(cs[0].at<double>({(cytnx_uint64)0, (cytnx_uint64)1}), 2.0, 1e-10);
    EXPECT_NEAR(cs[0].at<double>({(cytnx_uint64)1, (cytnx_uint64)0}), 3.0, 1e-10);
    EXPECT_NEAR(cs[0].at<double>({(cytnx_uint64)1, (cytnx_uint64)1}), 4.0, 1e-10);
  }

  /*=====test info=====
  describe: Double tensors + ComplexDouble alpha → result promoted to ComplexDouble.
    Real matrices scaled by a purely imaginary alpha produce purely imaginary output.
  input: A=I2, B=diag(2,3) (double), alpha=i, beta=0.
  Expected: result dtype == ComplexDouble; C[0][0]=2i, C[1][1]=3i, off-diagonal=0.
  ====================*/
  TEST(GemmBatch, TypePromotionDoubleTensorsComplexScalar) {
    using C128 = cytnx_complex128;
    int m = 2, n = 2, k = 2;
    Tensor A = zeros({(cytnx_uint64)m, (cytnx_uint64)k}, Type.Double, Device.cpu);
    Tensor B = zeros({(cytnx_uint64)k, (cytnx_uint64)n}, Type.Double, Device.cpu);
    Tensor C = zeros({(cytnx_uint64)m, (cytnx_uint64)n}, Type.Double, Device.cpu);
    A.at({(cytnx_uint64)0, (cytnx_uint64)0}) = 1.0;
    A.at({(cytnx_uint64)1, (cytnx_uint64)1}) = 1.0;  // identity
    B.at({(cytnx_uint64)0, (cytnx_uint64)0}) = 2.0;
    B.at({(cytnx_uint64)1, (cytnx_uint64)1}) = 3.0;

    std::vector<Tensor> as = {A}, bs = {B}, cs = {C};
    std::vector<Scalar> alphas = {Scalar(C128(0, 1))};  // alpha = i
    std::vector<Scalar> betas = {Scalar(C128(0, 0))};
    linalg::Gemm_Batch({(cytnx_int64)m}, {(cytnx_int64)n}, {(cytnx_int64)k}, alphas, as, bs, betas,
                       cs, 1, {1});

    EXPECT_EQ(cs[0].dtype(), (unsigned int)Type.ComplexDouble)
      << "double tensors with complex128 scalar must produce a ComplexDouble result";
    EXPECT_NEAR(cs[0].at<C128>({(cytnx_uint64)0, (cytnx_uint64)0}).imag(), 2.0, 1e-10);
    EXPECT_NEAR(cs[0].at<C128>({(cytnx_uint64)1, (cytnx_uint64)1}).imag(), 3.0, 1e-10);
    EXPECT_NEAR(cs[0].at<C128>({(cytnx_uint64)0, (cytnx_uint64)1}).real(), 0.0, 1e-10);
    EXPECT_NEAR(cs[0].at<C128>({(cytnx_uint64)0, (cytnx_uint64)1}).imag(), 0.0, 1e-10);
  }

#else  // !UNI_MKL

  // ═══════════════════════════════════════════════════════════════════════════════
  //  Without MKL: all four dispatch paths must throw std::logic_error
  // ═══════════════════════════════════════════════════════════════════════════════

  /*=====test info=====
  describe: Gemm_Batch_internal_{d,f,cd,cf} call cytnx_error_msg(true,...) when MKL is absent,
    which throws std::logic_error.  One test per dispatch path.
  ====================*/
  TEST(GemmBatchNoMKL, ThrowsForDouble) {
    Tensor A = zeros({2, 2}, Type.Double, Device.cpu);
    Tensor B = zeros({2, 2}, Type.Double, Device.cpu);
    Tensor C = zeros({2, 2}, Type.Double, Device.cpu);
    std::vector<Tensor> as = {A}, bs = {B}, cs = {C};
    std::vector<Scalar> alphas = {Scalar(1.0)}, betas = {Scalar(0.0)};
    EXPECT_THROW(linalg::Gemm_Batch({2}, {2}, {2}, alphas, as, bs, betas, cs, 1, {1}),
                 std::logic_error);
  }

  TEST(GemmBatchNoMKL, ThrowsForFloat) {
    Tensor A = zeros({2, 2}, Type.Float, Device.cpu);
    Tensor B = zeros({2, 2}, Type.Float, Device.cpu);
    Tensor C = zeros({2, 2}, Type.Float, Device.cpu);
    std::vector<Tensor> as = {A}, bs = {B}, cs = {C};
    std::vector<Scalar> alphas = {Scalar(1.0f)}, betas = {Scalar(0.0f)};
    EXPECT_THROW(linalg::Gemm_Batch({2}, {2}, {2}, alphas, as, bs, betas, cs, 1, {1}),
                 std::logic_error);
  }

  TEST(GemmBatchNoMKL, ThrowsForComplex128) {
    using C128 = cytnx_complex128;
    Tensor A = zeros({2, 2}, Type.ComplexDouble, Device.cpu);
    Tensor B = zeros({2, 2}, Type.ComplexDouble, Device.cpu);
    Tensor C = zeros({2, 2}, Type.ComplexDouble, Device.cpu);
    std::vector<Tensor> as = {A}, bs = {B}, cs = {C};
    std::vector<Scalar> alphas = {Scalar(C128(1, 0))}, betas = {Scalar(C128(0, 0))};
    EXPECT_THROW(linalg::Gemm_Batch({2}, {2}, {2}, alphas, as, bs, betas, cs, 1, {1}),
                 std::logic_error);
  }

  TEST(GemmBatchNoMKL, ThrowsForComplex64) {
    using C64 = cytnx_complex64;
    Tensor A = zeros({2, 2}, Type.ComplexFloat, Device.cpu);
    Tensor B = zeros({2, 2}, Type.ComplexFloat, Device.cpu);
    Tensor C = zeros({2, 2}, Type.ComplexFloat, Device.cpu);
    std::vector<Tensor> as = {A}, bs = {B}, cs = {C};
    std::vector<Scalar> alphas = {Scalar(C64(1, 0))}, betas = {Scalar(C64(0, 0))};
    EXPECT_THROW(linalg::Gemm_Batch({2}, {2}, {2}, alphas, as, bs, betas, cs, 1, {1}),
                 std::logic_error);
  }

#endif  // UNI_MKL

  // ═══════════════════════════════════════════════════════════════════════════════
  //  Error cases — dimension inconsistency (require User_debug = true)
  //
  //  linalg::Gemm_Batch validates tensor shapes inside `if (User_debug)` and throws
  //  std::logic_error before any BLAS call, so these tests work with or without MKL.
  // ═══════════════════════════════════════════════════════════════════════════════

  /*=====test info=====
  describe: A.cols (k) != B.rows violates the fundamental GEMM contraction dimension.
    A is 2×4, B is 5×3: B has 5 rows but A has only 4 columns.
  ====================*/
  TEST_F(GemmBatchDebugTest, ThrowsOnInnerDimMismatch) {
    Tensor A = zeros({2, 4}, Type.Double, Device.cpu);
    Tensor B = zeros({5, 3}, Type.Double, Device.cpu);  // wrong: rows should be 4
    Tensor C = zeros({2, 3}, Type.Double, Device.cpu);
    std::vector<Tensor> as = {A}, bs = {B}, cs = {C};
    std::vector<Scalar> alphas = {Scalar(1.0)}, betas = {Scalar(0.0)};
    EXPECT_THROW(linalg::Gemm_Batch({2}, {3}, {4}, alphas, as, bs, betas, cs, 1, {1}),
                 std::logic_error);
  }

  /*=====test info=====
  describe: A.rows != C.rows: the output tensor has the wrong number of rows.
    A is 2×3, C is 4×4: C has 4 rows but A has only 2.
  ====================*/
  TEST_F(GemmBatchDebugTest, ThrowsOnOutputRowMismatch) {
    Tensor A = zeros({2, 3}, Type.Double, Device.cpu);
    Tensor B = zeros({3, 4}, Type.Double, Device.cpu);
    Tensor C = zeros({4, 4}, Type.Double, Device.cpu);  // wrong: rows should be 2
    std::vector<Tensor> as = {A}, bs = {B}, cs = {C};
    std::vector<Scalar> alphas = {Scalar(1.0)}, betas = {Scalar(0.0)};
    EXPECT_THROW(linalg::Gemm_Batch({2}, {4}, {3}, alphas, as, bs, betas, cs, 1, {1}),
                 std::logic_error);
  }

  /*=====test info=====
  describe: B.cols != C.cols: the output tensor has the wrong number of columns.
    B is 3×4, C is 2×7: C has 7 columns but B has only 4.
  ====================*/
  TEST_F(GemmBatchDebugTest, ThrowsOnOutputColMismatch) {
    Tensor A = zeros({2, 3}, Type.Double, Device.cpu);
    Tensor B = zeros({3, 4}, Type.Double, Device.cpu);
    Tensor C = zeros({2, 7}, Type.Double, Device.cpu);  // wrong: cols should be 4
    std::vector<Tensor> as = {A}, bs = {B}, cs = {C};
    std::vector<Scalar> alphas = {Scalar(1.0)}, betas = {Scalar(0.0)};
    EXPECT_THROW(linalg::Gemm_Batch({2}, {4}, {3}, alphas, as, bs, betas, cs, 1, {1}),
                 std::logic_error);
  }

  /*=====test info=====
  describe: group_size array length does not match group_count.
    group_count=2 but only 1 entry in group_size.
  ====================*/
  TEST_F(GemmBatchDebugTest, ThrowsOnGroupCountMismatch) {
    Tensor A = zeros({2, 2}, Type.Double, Device.cpu);
    Tensor B = zeros({2, 2}, Type.Double, Device.cpu);
    Tensor C = zeros({2, 2}, Type.Double, Device.cpu);
    std::vector<Tensor> as = {A}, bs = {B}, cs = {C};
    std::vector<Scalar> alphas = {Scalar(1.0)}, betas = {Scalar(0.0)};
    // group_count=2 but group_size has only 1 element
    EXPECT_THROW(linalg::Gemm_Batch({2}, {2}, {2}, alphas, as, bs, betas, cs, 2, {1}),
                 std::logic_error);
  }

  /*=====test info=====
  describe: Number of tensors in as/bs/cs does not equal the sum of group_sizes.
    group_count=1, group_size={3} implies 3 matrices, but only 1 tensor is provided.
  ====================*/
  TEST_F(GemmBatchDebugTest, ThrowsOnTensorCountMismatch) {
    Tensor A = zeros({2, 2}, Type.Double, Device.cpu);
    Tensor B = zeros({2, 2}, Type.Double, Device.cpu);
    Tensor C = zeros({2, 2}, Type.Double, Device.cpu);
    std::vector<Tensor> as = {A}, bs = {B}, cs = {C};
    std::vector<Scalar> alphas = {Scalar(1.0)}, betas = {Scalar(0.0)};
    // group_size={3} → 3 matrices expected, but we only provided 1
    EXPECT_THROW(linalg::Gemm_Batch({2}, {2}, {2}, alphas, as, bs, betas, cs, 1, {3}),
                 std::logic_error);
  }

  // ═══════════════════════════════════════════════════════════════════════════════
  //  Error cases — invalid scalar types (unconditional, no User_debug needed)
  //
  //  Gemm_Batch rejects non-floating-point and Void scalars before any BLAS call,
  //  so these tests work with or without MKL and regardless of User_debug.
  // ═══════════════════════════════════════════════════════════════════════════════

  /*=====test info=====
  describe: alpha with an integer dtype (Int64, dtype=5 > 4) is rejected unconditionally.
    Only (complex/real)(double/float) scalars are accepted.
  ====================*/
  TEST(GemmBatchError, ThrowsOnIntegerAlpha) {
    Tensor A = zeros({2, 2}, Type.Double, Device.cpu);
    Tensor B = zeros({2, 2}, Type.Double, Device.cpu);
    Tensor C = zeros({2, 2}, Type.Double, Device.cpu);
    std::vector<Tensor> as = {A}, bs = {B}, cs = {C};
    std::vector<Scalar> alphas = {Scalar(cytnx_int64(1))};  // integer type: dtype=5 > 4
    std::vector<Scalar> betas = {Scalar(0.0)};
    EXPECT_THROW(linalg::Gemm_Batch({2}, {2}, {2}, alphas, as, bs, betas, cs, 1, {1}),
                 std::logic_error);
  }

  /*=====test info=====
  describe: beta with a Void dtype (default-constructed Scalar) is rejected unconditionally.
  ====================*/
  TEST(GemmBatchError, ThrowsOnVoidBeta) {
    Tensor A = zeros({2, 2}, Type.Double, Device.cpu);
    Tensor B = zeros({2, 2}, Type.Double, Device.cpu);
    Tensor C = zeros({2, 2}, Type.Double, Device.cpu);
    std::vector<Tensor> as = {A}, bs = {B}, cs = {C};
    std::vector<Scalar> alphas = {Scalar(1.0)};
    std::vector<Scalar> betas = {Scalar()};  // default-constructed → Void dtype
    EXPECT_THROW(linalg::Gemm_Batch({2}, {2}, {2}, alphas, as, bs, betas, cs, 1, {1}),
                 std::logic_error);
  }

}  // namespace cytnx
