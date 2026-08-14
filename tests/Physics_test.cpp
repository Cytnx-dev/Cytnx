#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "Device.hpp"
#include "Physics.hpp"
#include "Type.hpp"
#include "linalg.hpp"

// physics::spin() builds the spin-S matrices. The 'y' component used to form its entries as
// `cytnx_complex128(0, 1) * pow(..., 0.5) / 2`, which depended on the namespace-scope
// complex/builtin operators removed in #1003; it now constructs the imaginary entry directly.
// The 'x' component moved from pow(..., 0.5) to std::sqrt at the same time.
//
// The expected values below are the textbook hbar = 1 spin matrices, written out independently of
// how Physics.cpp computes them.
namespace cytnx {
  namespace test {
    namespace {

      constexpr double kTol = 1e-14;

      void ExpectElement(const Tensor& t, cytnx_uint64 row, cytnx_uint64 col, double real,
                         double imag) {
        const cytnx_complex128 value = t.at<cytnx_complex128>({row, col});
        EXPECT_NEAR(value.real(), real, kTol) << "at (" << row << ", " << col << ")";
        EXPECT_NEAR(value.imag(), imag, kTol) << "at (" << row << ", " << col << ")";
      }

      TEST(PhysicsSpin, HalfComponents) {
        // S = 1/2: S_a = sigma_a / 2.
        Tensor sx = physics::spin(0.5, "x");
        ASSERT_EQ(sx.dtype(), Type.ComplexDouble);
        ASSERT_EQ(sx.shape(), std::vector<cytnx_uint64>({2, 2}));
        ExpectElement(sx, 0, 0, 0.0, 0.0);
        ExpectElement(sx, 0, 1, 0.5, 0.0);
        ExpectElement(sx, 1, 0, 0.5, 0.0);
        ExpectElement(sx, 1, 1, 0.0, 0.0);

        Tensor sy = physics::spin(0.5, "y");
        ASSERT_EQ(sy.dtype(), Type.ComplexDouble);
        ASSERT_EQ(sy.shape(), std::vector<cytnx_uint64>({2, 2}));
        ExpectElement(sy, 0, 0, 0.0, 0.0);
        ExpectElement(sy, 0, 1, 0.0, -0.5);
        ExpectElement(sy, 1, 0, 0.0, 0.5);
        ExpectElement(sy, 1, 1, 0.0, 0.0);

        Tensor sz = physics::spin(0.5, "z");
        ASSERT_EQ(sz.shape(), std::vector<cytnx_uint64>({2, 2}));
        ExpectElement(sz, 0, 0, 0.5, 0.0);
        ExpectElement(sz, 0, 1, 0.0, 0.0);
        ExpectElement(sz, 1, 0, 0.0, 0.0);
        ExpectElement(sz, 1, 1, -0.5, 0.0);
      }

      TEST(PhysicsSpin, OneComponents) {
        // S = 1: the off-diagonal entries are 1/sqrt(2), which is irrational -- exactly the value
        // the pow(x, 0.5) -> std::sqrt change touches.
        const double off = 1.0 / std::sqrt(2.0);

        Tensor sx = physics::spin(1.0, "x");
        ASSERT_EQ(sx.shape(), std::vector<cytnx_uint64>({3, 3}));
        ExpectElement(sx, 0, 1, off, 0.0);
        ExpectElement(sx, 1, 0, off, 0.0);
        ExpectElement(sx, 1, 2, off, 0.0);
        ExpectElement(sx, 2, 1, off, 0.0);
        ExpectElement(sx, 0, 0, 0.0, 0.0);
        ExpectElement(sx, 0, 2, 0.0, 0.0);
        ExpectElement(sx, 2, 0, 0.0, 0.0);

        Tensor sy = physics::spin(1.0, "y");
        ASSERT_EQ(sy.shape(), std::vector<cytnx_uint64>({3, 3}));
        ExpectElement(sy, 0, 1, 0.0, -off);
        ExpectElement(sy, 1, 0, 0.0, off);
        ExpectElement(sy, 1, 2, 0.0, -off);
        ExpectElement(sy, 2, 1, 0.0, off);
        ExpectElement(sy, 0, 0, 0.0, 0.0);
        ExpectElement(sy, 0, 2, 0.0, 0.0);
        ExpectElement(sy, 2, 0, 0.0, 0.0);

        Tensor sz = physics::spin(1.0, "z");
        ASSERT_EQ(sz.shape(), std::vector<cytnx_uint64>({3, 3}));
        ExpectElement(sz, 0, 0, 1.0, 0.0);
        ExpectElement(sz, 1, 1, 0.0, 0.0);
        ExpectElement(sz, 2, 2, -1.0, 0.0);
      }

      TEST(PhysicsSpin, CommutatorSatisfiesSu2Algebra) {
        // An independent algebraic check that does not reuse the elementwise formulas:
        // [Sx, Sy] = i Sz must hold for every S.
        for (double s : {0.5, 1.0, 1.5, 2.0}) {
          Tensor sx = physics::spin(s, "x");
          Tensor sy = physics::spin(s, "y");
          Tensor sz = physics::spin(s, "z");
          Tensor commutator = linalg::Matmul(sx, sy) - linalg::Matmul(sy, sx);

          const cytnx_uint64 dim = sx.shape()[0];
          for (cytnx_uint64 i = 0; i < dim; ++i) {
            for (cytnx_uint64 j = 0; j < dim; ++j) {
              // i * Sz, with Sz real diagonal, is purely imaginary.
              const cytnx_complex128 expect(0.0, sz.at<cytnx_complex128>({i, j}).real());
              const cytnx_complex128 got = commutator.at<cytnx_complex128>({i, j});
              EXPECT_NEAR(got.real(), expect.real(), kTol)
                << "S=" << s << " at (" << i << ", " << j << ")";
              EXPECT_NEAR(got.imag(), expect.imag(), kTol)
                << "S=" << s << " at (" << i << ", " << j << ")";
            }
          }
        }
      }

      TEST(PhysicsSpin, RejectsNonHalfIntegerSpin) {
        EXPECT_THROW(physics::spin(0.3, "z"), cytnx::error);
        EXPECT_THROW(physics::spin(0.25, "x"), cytnx::error);
        // S below 1/2 is rejected outright.
        EXPECT_THROW(physics::spin(0.0, "z"), cytnx::error);
      }

    }  // namespace
  }  // namespace test
}  // namespace cytnx
