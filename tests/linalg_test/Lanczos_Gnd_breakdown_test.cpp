#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "cytnx.hpp"
#include "linalg_test.h"

// Regression tests for #943 and #903: linalg::Lanczos(method = "Gnd") on operators whose Krylov
// space is exhausted early (1x1 operators, a start vector that is an exact eigenvector, scalar
// multiples of the identity) or is as large as the whole vector space. Every expected eigenvalue
// and eigenvector below is known in closed form, and norms, overlaps and residuals are evaluated
// with plain std::complex arithmetic rather than with Cytnx linear algebra.

namespace cytnx {
  namespace test {
    namespace {

      using Complex = std::complex<double>;
      using Vector = std::vector<Complex>;
      using Matrix = std::vector<Vector>;

      enum class VecKind { kTensor, kUniTensor };

      Tensor MakeVector(const Vector& values, unsigned int dtype) {
        const cytnx_uint64 n = values.size();
        if (Type.is_complex(dtype)) {
          Tensor out = zeros({n}, Type.ComplexDouble);
          for (cytnx_uint64 i = 0; i < n; i++) out.at<cytnx_complex128>({i}) = values[i];
          return out.astype(dtype);
        }
        Tensor out = zeros({n}, Type.Double);
        for (cytnx_uint64 i = 0; i < n; i++) out.at<cytnx_double>({i}) = values[i].real();
        return out.astype(dtype);
      }

      Tensor MakeMatrix(const Matrix& values, unsigned int dtype) {
        const cytnx_uint64 n = values.size();
        if (Type.is_complex(dtype)) {
          Tensor out = zeros({n, n}, Type.ComplexDouble);
          for (cytnx_uint64 i = 0; i < n; i++)
            for (cytnx_uint64 j = 0; j < n; j++) out.at<cytnx_complex128>({i, j}) = values[i][j];
          return out.astype(dtype);
        }
        Tensor out = zeros({n, n}, Type.Double);
        for (cytnx_uint64 i = 0; i < n; i++)
          for (cytnx_uint64 j = 0; j < n; j++) out.at<cytnx_double>({i, j}) = values[i][j].real();
        return out.astype(dtype);
      }

      Vector ToStdVector(const Tensor& t) {
        Tensor c = t.astype(Type.ComplexDouble).contiguous();
        Vector out(c.shape()[0]);
        for (cytnx_uint64 i = 0; i < out.size(); i++) out[i] = c.at<cytnx_complex128>({i});
        return out;
      }

      // A dense matrix acting on a rank-1 Tensor or on a rank-1 dense UniTensor, counting how often
      // it is applied.
      class DenseMatOp : public LinOp {
       public:
        explicit DenseMatOp(const Tensor& mat)
            : LinOp("mv", mat.shape()[0], mat.dtype(), mat.device()), mat_(mat) {}

        Tensor matvec(const Tensor& v) override {
          ++matvec_count;
          return linalg::Dot(mat_, v);
        }

        UniTensor matvec(const UniTensor& v) override {
          ++matvec_count;
          return UniTensor(linalg::Dot(mat_, v.get_block_()));
        }

        int matvec_count = 0;

       private:
        Tensor mat_;
      };

      struct GndResult {
        double energy = 0.0;
        cytnx_uint64 num_outputs = 0;
        Vector vec;  // empty when no eigenvector was requested
        int matvec_count = 0;
      };

      GndResult RunGnd(const Matrix& mat, const Vector& start, unsigned int dtype, VecKind kind,
                       bool is_V = true, double cvg_crit = 1.0e-14, unsigned int maxiter = 20) {
        DenseMatOp op(MakeMatrix(mat, dtype));
        Tensor tin = MakeVector(start, dtype);
        GndResult res;
        Tensor vec;
        if (kind == VecKind::kTensor) {
          std::vector<Tensor> out =
            linalg::Lanczos(&op, tin, "Gnd", cvg_crit, maxiter, 1, is_V, false, 0, false);
          res.num_outputs = out.size();
          res.energy = double(out[0].item().real());
          if (is_V) vec = out[1];
        } else {
          std::vector<UniTensor> out = linalg::Lanczos(&op, UniTensor(tin), "Gnd", cvg_crit,
                                                       maxiter, 1, is_V, false, 0, false);
          res.num_outputs = out.size();
          res.energy = double(out[0].get_block_().item().real());
          if (is_V) vec = out[1].get_block_();
        }
        if (is_V) res.vec = ToStdVector(vec);
        res.matvec_count = op.matvec_count;
        return res;
      }

      double NormOf(const Vector& v) {
        double sum = 0.0;
        for (const Complex& x : v) sum += std::norm(x);
        return std::sqrt(sum);
      }

      // || mat v - lambda v ||
      double Residual(const Matrix& mat, const Vector& v, double lambda) {
        Vector r(v.size());
        for (size_t i = 0; i < v.size(); i++) {
          r[i] = -lambda * v[i];
          for (size_t j = 0; j < v.size(); j++) r[i] += mat[i][j] * v[j];
        }
        return NormOf(r);
      }

      // |<a|b>| / (|a| |b|): 1 exactly when the two vectors are parallel.
      double Overlap(const Vector& a, const Vector& b) {
        Complex dot = 0.0;
        for (size_t i = 0; i < a.size(); i++) dot += std::conj(a[i]) * b[i];
        return std::abs(dot) / (NormOf(a) * NormOf(b));
      }

      double ToleranceFor(unsigned int dtype) {
        return (dtype == Type.Float || dtype == Type.ComplexFloat) ? 1.0e-5 : 1.0e-12;
      }

      void ExpectEigenpair(const GndResult& res, const Matrix& mat, double expected_energy,
                           const Vector& expected_vec, unsigned int dtype) {
        const double tol = ToleranceFor(dtype);
        EXPECT_EQ(res.num_outputs, 2);
        EXPECT_NEAR(res.energy, expected_energy, tol);
        ASSERT_EQ(res.vec.size(), expected_vec.size());
        EXPECT_NEAR(NormOf(res.vec), 1.0, tol);
        EXPECT_LT(Residual(mat, res.vec, expected_energy), tol);
        EXPECT_NEAR(Overlap(expected_vec, res.vec), 1.0, tol);
      }

      std::string ParamName(
        const ::testing::TestParamInfo<std::tuple<VecKind, unsigned int>>& info) {
        const std::string kind =
          std::get<0>(info.param) == VecKind::kTensor ? "Tensor" : "UniTensor";
        const unsigned int dtype = std::get<1>(info.param);
        std::string dtype_name = "Double";
        if (dtype == Type.ComplexDouble) dtype_name = "ComplexDouble";
        if (dtype == Type.ComplexFloat) dtype_name = "ComplexFloat";
        if (dtype == Type.Float) dtype_name = "Float";
        return kind + "_" + dtype_name;
      }

      class LanczosGndBreakdown
          : public ::testing::TestWithParam<std::tuple<VecKind, unsigned int>> {
       protected:
        VecKind kind() const { return std::get<0>(GetParam()); }
        unsigned int dtype() const { return std::get<1>(GetParam()); }
      };

      // H = [3], v = [1]: the Krylov space is one-dimensional.
      TEST_P(LanczosGndBreakdown, Scalar1x1) {
        const Matrix mat = {{3.0}};
        ExpectEigenpair(RunGnd(mat, {1.0}, dtype(), kind()), mat, 3.0, {1.0}, dtype());
      }

      TEST_P(LanczosGndBreakdown, Scalar1x1EnergyOnly) {
        const Matrix mat = {{3.0}};
        GndResult res = RunGnd(mat, {2.0}, dtype(), kind(), false);
        EXPECT_EQ(res.num_outputs, 1);
        EXPECT_NEAR(res.energy, 3.0, ToleranceFor(dtype()));
        EXPECT_EQ(res.matvec_count, 1);
      }

      // The start vector is the ground state itself.
      TEST_P(LanczosGndBreakdown, DiagonalStartAtGroundState2D) {
        const Matrix mat = {{1.0, 0.0}, {0.0, 2.0}};
        ExpectEigenpair(RunGnd(mat, {1.0, 0.0}, dtype(), kind()), mat, 1.0, {1.0, 0.0}, dtype());
      }

      TEST_P(LanczosGndBreakdown, DiagonalStartAtGroundState3D) {
        const Matrix mat = {{1.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 3.0}};
        ExpectEigenpair(RunGnd(mat, {1.0, 0.0, 0.0}, dtype(), kind()), mat, 1.0, {1.0, 0.0, 0.0},
                        dtype());
      }

      // The start vector is an eigenvector that is not the ground state. Its Krylov space is the
      // span of that one vector, so the lowest eigenpair inside it is the start vector itself: a
      // Krylov method cannot leave the invariant subspace it starts in.
      TEST_P(LanczosGndBreakdown, DiagonalStartAtExcitedEigenvector) {
        const Matrix mat = {{1.0, 0.0}, {0.0, 2.0}};
        ExpectEigenpair(RunGnd(mat, {0.0, 1.0}, dtype(), kind()), mat, 2.0, {0.0, 1.0}, dtype());
      }

      // H = 3 I: every vector is an eigenvector, so the result is the normalized start vector.
      TEST_P(LanczosGndBreakdown, ScalarIdentity2D) {
        const Matrix mat = {{3.0, 0.0}, {0.0, 3.0}};
        ExpectEigenpair(RunGnd(mat, {1.0, 1.0}, dtype(), kind()), mat, 3.0, {1.0, 1.0}, dtype());
      }

      TEST_P(LanczosGndBreakdown, ScalarIdentity3D) {
        const Matrix mat = {{2.0, 0.0, 0.0}, {0.0, 2.0, 0.0}, {0.0, 0.0, 2.0}};
        ExpectEigenpair(RunGnd(mat, {1.0, 1.0, 1.0}, dtype(), kind()), mat, 2.0, {1.0, 1.0, 1.0},
                        dtype());
      }

      // H = diag(1, 0), v = (1, 1): the Krylov space is the whole space after two steps.
      TEST_P(LanczosGndBreakdown, RankOneProjector) {
        const Matrix mat = {{1.0, 0.0}, {0.0, 0.0}};
        ExpectEigenpair(RunGnd(mat, {1.0, 1.0}, dtype(), kind()), mat, 0.0, {0.0, 1.0}, dtype());
      }

      // H = [[2, 1], [1, 2]] has eigenvalues 1 and 3; the ground state is (1, -1) / sqrt(2).
      TEST_P(LanczosGndBreakdown, Symmetric2x2) {
        const Matrix mat = {{2.0, 1.0}, {1.0, 2.0}};
        ExpectEigenpair(RunGnd(mat, {1.0, 0.0}, dtype(), kind()), mat, 1.0, {1.0, -1.0}, dtype());
      }

      // The 3-site open chain [[2, -1, 0], [-1, 2, -1], [0, -1, 2]] has eigenvalues 2 - sqrt(2), 2
      // and 2 + sqrt(2); the ground state is (1, sqrt(2), 1) / 2.
      TEST_P(LanczosGndBreakdown, Tridiagonal3x3) {
        const Matrix mat = {{2.0, -1.0, 0.0}, {-1.0, 2.0, -1.0}, {0.0, -1.0, 2.0}};
        ExpectEigenpair(RunGnd(mat, {1.0, 0.0, 0.0}, dtype(), kind()), mat, 2.0 - std::sqrt(2.0),
                        {1.0, std::sqrt(2.0), 1.0}, dtype());
      }

      // A vector space of dimension n holds at most n Krylov vectors, so at most n applications of
      // the operator are needed however small the requested tolerance is.
      TEST_P(LanczosGndBreakdown, KrylovDimensionCappedAtVectorDimension) {
        const Matrix mat = {
          {1.0, 0.0, 0.0, 0.0}, {0.0, 2.0, 0.0, 0.0}, {0.0, 0.0, 3.0, 0.0}, {0.0, 0.0, 0.0, 4.0}};
        const Vector start = {1.0, 1.0, 1.0, 1.0};
        GndResult energy_only = RunGnd(mat, start, dtype(), kind(), false, 1.0e-300, 20);
        EXPECT_NEAR(energy_only.energy, 1.0, ToleranceFor(dtype()));
        EXPECT_LE(energy_only.matvec_count, 4);
        ExpectEigenpair(RunGnd(mat, start, dtype(), kind(), true, 1.0e-300, 20), mat, 1.0,
                        {1.0, 0.0, 0.0, 0.0}, dtype());
      }

      // A zero start vector has no direction to normalize; it must be rejected, not divided by.
      TEST_P(LanczosGndBreakdown, ZeroStartVectorIsRejected) {
        const Matrix mat = {{1.0, 0.0}, {0.0, 2.0}};
        EXPECT_THROW(RunGnd(mat, {0.0, 0.0}, dtype(), kind()), cytnx::error);
      }

      // A Float operator cannot resolve energy changes below 1e-7: a smaller CvgCrit is raised to
      // 1e-7, and the caller must be told so. Double operators and tolerances of 1e-7 or more are
      // left alone.
      TEST_P(LanczosGndBreakdown, FloatCvgCritFloorWarns) {
        const Matrix mat = {{2.0, 1.0}, {1.0, 2.0}};
        const bool is_float = dtype() == Type.Float || dtype() == Type.ComplexFloat;
        testing::internal::CaptureStderr();
        RunGnd(mat, {1.0, 0.0}, dtype(), kind(), true, 1.0e-8);
        const std::string below_floor = testing::internal::GetCapturedStderr();
        EXPECT_EQ(below_floor.find("CvgCrit") != std::string::npos, is_float) << below_floor;

        testing::internal::CaptureStderr();
        RunGnd(mat, {1.0, 0.0}, dtype(), kind(), true, 1.0e-6);
        const std::string above_floor = testing::internal::GetCapturedStderr();
        EXPECT_EQ(above_floor.find("CvgCrit"), std::string::npos) << above_floor;
      }

      INSTANTIATE_TEST_SUITE_P(
        AllPaths, LanczosGndBreakdown,
        ::testing::Combine(::testing::Values(VecKind::kTensor, VecKind::kUniTensor),
                           ::testing::Values(Type.Double, Type.Float, Type.ComplexDouble,
                                             Type.ComplexFloat)),
        ParamName);

      class LanczosGndComplexHermitian : public LanczosGndBreakdown {};

      // H = [[0, i], [-i, 0]] has eigenvalues -1 and +1; the ground state is (1, i) / sqrt(2).
      TEST_P(LanczosGndComplexHermitian, PauliY) {
        const Complex i(0.0, 1.0);
        const Matrix mat = {{0.0, i}, {-i, 0.0}};
        ExpectEigenpair(RunGnd(mat, {1.0, 0.0}, dtype(), kind()), mat, -1.0, {1.0, i}, dtype());
      }

      TEST_P(LanczosGndComplexHermitian, PauliYStartAtGroundState) {
        const Complex i(0.0, 1.0);
        const Matrix mat = {{0.0, i}, {-i, 0.0}};
        ExpectEigenpair(RunGnd(mat, {1.0, i}, dtype(), kind()), mat, -1.0, {1.0, i}, dtype());
      }

      // H = [[2, 1 - i], [1 + i, 3]] has trace 5 and determinant 4, hence eigenvalues 1 and 4; the
      // ground state is (-1 + i, 1) / sqrt(3).
      TEST_P(LanczosGndComplexHermitian, General2x2) {
        const Matrix mat = {{2.0, Complex(1.0, -1.0)}, {Complex(1.0, 1.0), 3.0}};
        ExpectEigenpair(RunGnd(mat, {1.0, 0.0}, dtype(), kind()), mat, 1.0,
                        {Complex(-1.0, 1.0), 1.0}, dtype());
      }

      INSTANTIATE_TEST_SUITE_P(
        AllPaths, LanczosGndComplexHermitian,
        ::testing::Combine(::testing::Values(VecKind::kTensor, VecKind::kUniTensor),
                           ::testing::Values(Type.ComplexDouble, Type.ComplexFloat)),
        ParamName);

      // U(1)-symmetric operator with the charge-0 block [[2, 1], [1, 2]] (eigenvalues 1 and 3) and
      // the charge-1 block [5], acting on a ket with three stored elements: (k0, k1) in the
      // charge-0 sector and k2 in the charge-1 sector.
      class BlockDiagonalOp : public LinOp {
       public:
        BlockDiagonalOp() : LinOp("mv", 3) {
          Bond in = Bond(BD_IN, {Qs(0), Qs(1)}, {2, 1});
          Bond out = Bond(BD_OUT, {Qs(0), Qs(1)}, {2, 1});
          h_ = UniTensor({in, out});
          h_.at({0, 0}) = 2.0;
          h_.at({0, 1}) = 1.0;
          h_.at({1, 0}) = 1.0;
          h_.at({1, 1}) = 2.0;
          h_.at({2, 2}) = 5.0;
          h_.relabel_({"a", "b"});
        }

        UniTensor matvec(const UniTensor& psi) override {
          ++matvec_count;
          UniTensor out = h_.contract(psi);
          out.relabel_({"b", "c"});
          return out;
        }

        static UniTensor MakeKet(double k0, double k1, double k2) {
          Bond in = Bond(BD_IN, {Qs(0), Qs(1)}, {2, 1});
          Bond out = Bond(BD_OUT, {Qs(0), Qs(1)}, {1, 1});
          UniTensor ket = UniTensor({in, out});
          ket.at({0, 0}) = k0;
          ket.at({1, 0}) = k1;
          ket.at({2, 1}) = k2;
          ket.relabel_({"b", "c"});
          return ket;
        }

        static Vector Elements(const UniTensor& ket) {
          UniTensor k = ket.clone();
          return {double(k.at({0, 0}).real()), double(k.at({1, 0}).real()),
                  double(k.at({2, 1}).real())};
        }

        int matvec_count = 0;

       private:
        UniTensor h_;
      };

      const Matrix kBlockDiagonalMatrix = {{2.0, 1.0, 0.0}, {1.0, 2.0, 0.0}, {0.0, 0.0, 5.0}};

      void ExpectBlockEigenpair(const std::vector<UniTensor>& eigs, double expected_energy,
                                const Vector& expected_vec) {
        ASSERT_EQ(eigs.size(), 2);
        UniTensor energy = eigs[0];  // non-const handle: item() on a const Tensor is read-only
        EXPECT_NEAR(double(energy.get_block_().item().real()), expected_energy, 1.0e-12);
        EXPECT_EQ(eigs[1].uten_type(), UTenType.Block);
        const Vector vec = BlockDiagonalOp::Elements(eigs[1]);
        EXPECT_NEAR(NormOf(vec), 1.0, 1.0e-12);
        EXPECT_LT(Residual(kBlockDiagonalMatrix, vec, expected_energy), 1.0e-12);
        EXPECT_NEAR(Overlap(expected_vec, vec), 1.0, 1.0e-12);
      }

      TEST(LanczosGndBreakdownBlock, StartAtGroundState) {
        BlockDiagonalOp op;
        auto eigs = linalg::Lanczos(&op, BlockDiagonalOp::MakeKet(1.0, -1.0, 0.0), "Gnd", 1.0e-14,
                                    20, 1, true, false, 0, false);
        ExpectBlockEigenpair(eigs, 1.0, {1.0, -1.0, 0.0});
      }

      // The start vector lives entirely in the charge-1 sector, whose only eigenvalue is 5.
      TEST(LanczosGndBreakdownBlock, StartInOneSector) {
        BlockDiagonalOp op;
        auto eigs = linalg::Lanczos(&op, BlockDiagonalOp::MakeKet(0.0, 0.0, 1.0), "Gnd", 1.0e-14,
                                    20, 1, true, false, 0, false);
        ExpectBlockEigenpair(eigs, 5.0, {0.0, 0.0, 1.0});
      }

      // The Krylov space of (1, 0, 1) is the whole three-dimensional space, so three applications
      // of the operator suffice.
      TEST(LanczosGndBreakdownBlock, KrylovDimensionCappedAtStoredElements) {
        BlockDiagonalOp op;
        auto eigs = linalg::Lanczos(&op, BlockDiagonalOp::MakeKet(1.0, 0.0, 1.0), "Gnd", 1.0e-300,
                                    20, 1, true, false, 0, false);
        ExpectBlockEigenpair(eigs, 1.0, {1.0, -1.0, 0.0});
        EXPECT_LE(op.matvec_count, 3);
      }

      class CountingFermiAdaOp : public FermiAdaOp {
       public:
        using FermiAdaOp::FermiAdaOp;
        UniTensor matvec(const UniTensor& psi) override {
          ++matvec_count;
          return FermiAdaOp::matvec(psi);
        }
        int matvec_count = 0;
      };

      // Fermionic operator A^dag A started from its dense ground state. The first residual is at
      // roundoff level rather than exactly zero, and one application of the operator is all an
      // eigenvector needs.
      TEST(LanczosGndBreakdownFermionic, StartAtGroundState) {
        UniTensor A = make_ferm_A();
        auto low = ferm_dense_lowest(A, 1);
        UniTensor v0 = low[0].second.clone();
        CountingFermiAdaOp op(A, ferm_ket_nx(v0));
        auto eigs = linalg::Lanczos(&op, v0, "Gnd", 1.0e-12, 1000, 1, true);
        expect_lowest_states(A, eigs, low, 1.0e-7);
        EXPECT_EQ(op.matvec_count, 1);
      }

      // #903: a failing LAPACK call inside Tridiag must raise an error that names the LAPACK
      // routine and its INFO value, for both real dtypes and for either value of throw_excp.
      // LAPACKE rejects a NaN on the diagonal with INFO = -4.
      void ExpectTridiagLapackError(unsigned int dtype, bool throw_excp,
                                    const std::string& routine) {
        Tensor diag = zeros({2}, dtype);
        Tensor sub_diag = zeros({1}, dtype);
        if (dtype == Type.Double) {
          diag.at<cytnx_double>({0}) = std::numeric_limits<double>::quiet_NaN();
        } else {
          diag.at<cytnx_float>({0}) = std::numeric_limits<float>::quiet_NaN();
        }
        try {
          linalg::Tridiag(diag, sub_diag, true, true, throw_excp);
          FAIL() << "Tridiag accepted a NaN diagonal";
        } catch (const std::logic_error& err) {
          const std::string msg = err.what();
          EXPECT_NE(msg.find(routine), std::string::npos) << msg;
          EXPECT_NE(msg.find("INFO ="), std::string::npos) << msg;
          EXPECT_NE(msg.find("-4"), std::string::npos) << msg;
        }
      }

      TEST(TridiagLapackError, Double) { ExpectTridiagLapackError(Type.Double, false, "dstev"); }
      TEST(TridiagLapackError, DoubleThrowExcp) {
        ExpectTridiagLapackError(Type.Double, true, "dstev");
      }
      TEST(TridiagLapackError, Float) { ExpectTridiagLapackError(Type.Float, false, "sstev"); }
      TEST(TridiagLapackError, FloatThrowExcp) {
        ExpectTridiagLapackError(Type.Float, true, "sstev");
      }

    }  // namespace
  }  // namespace test
}  // namespace cytnx
