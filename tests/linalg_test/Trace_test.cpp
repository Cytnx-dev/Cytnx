#include <vector>

#include <gtest/gtest.h>

#include "linalg.hpp"
#include "random.hpp"
#include "Tensor.hpp"
#include "test_tools.h"
#include "Type.hpp"

namespace cytnx {
  namespace test {
    namespace {

      static Tensor ZeroExtentTensor(const std::vector<cytnx_uint64>& shape, unsigned int dtype) {
        return Tensor(shape, dtype, Device.cpu);
      }

      // The strided in-place trace must agree with the trace of a fully materialized
      // contiguous clone of the same tensor (which is the layout the old code always
      // assumed). Pairing both via the same public API isolates the layout choice.
      static Tensor ContiguousReferenceTrace(const Tensor& t, cytnx_uint64 a, cytnx_uint64 b) {
        return linalg::Trace(t.contiguous(), a, b);
      }

      TEST(LinalgTraceTest, PermutedRank3MatchesContiguous) {
        auto t = random::random_tensor({4, 3, 4}, -1.0, 1.0, Device.cpu, 0, Type.Double);
        auto p = t.permute({2, 1, 0});  // shape {4, 3, 4} but non-contiguous
        ASSERT_FALSE(p.is_contiguous());
        auto strided = linalg::Trace(p, 0, 2);
        auto reference = ContiguousReferenceTrace(p, 0, 2);
        EXPECT_TRUE(test::AreNearlyEqTensor(strided, reference, 1e-12));
      }

      TEST(LinalgTraceTest, PermutedTracesAcrossRanksAndDtypes) {
        struct Case {
          std::vector<cytnx_uint64> shape;
          std::vector<cytnx_uint64> perm;
          cytnx_uint64 ax1, ax2;
        };
        // Each case picks two equal-sized logical axes after a non-trivial permute.
        const std::vector<Case> cases = {
          // rank 3
          {{4, 3, 4}, {1, 0, 2}, 1, 2},
          // rank 4 with non-adjacent traced axes (permuted shape = {4, 4, 5, 3})
          {{4, 3, 4, 5}, {2, 0, 3, 1}, 0, 1},
          // rank 5
          {{3, 4, 3, 2, 3}, {4, 1, 0, 3, 2}, 0, 2},
        };
        for (unsigned int dtype : {Type.Double, Type.ComplexDouble, Type.Int32}) {
          for (const auto& c : cases) {
            auto t = random::random_tensor(c.shape, -2.0, 2.0, Device.cpu, 0, dtype);
            auto p = t.permute(c.perm);
            EXPECT_FALSE(p.is_contiguous());
            auto strided = linalg::Trace(p, c.ax1, c.ax2);
            auto reference = ContiguousReferenceTrace(p, c.ax1, c.ax2);
            EXPECT_TRUE(test::AreNearlyEqTensor(strided, reference, 1e-10))
              << "dtype=" << dtype << " rank=" << c.shape.size();
          }
        }
      }

      TEST(LinalgTraceTest, Rank2Path) {
        // Exercises _trace_2d / cuTrace_2d_kernel (the 2d branch is only taken when
        // every remaining axis has been traced away).
        auto t = random::random_tensor({6, 6}, -1.0, 1.0, Device.cpu, 0, Type.Double);
        auto p = t.permute({1, 0});
        ASSERT_FALSE(p.is_contiguous());
        auto out_p = linalg::Trace(p, 0, 1);
        auto out_c = linalg::Trace(t, 0, 1);  // tr(A) == tr(A^T)
        ASSERT_TRUE(out_p.is_scalar());
        ASSERT_TRUE(out_c.is_scalar());
        EXPECT_TRUE(test::AreNearlyEqTensor(out_p, out_c, 1e-12));
      }

      TEST(LinalgTraceTest, OutputRankIsInputMinusTwo) {
        // tr(rank-N) -> rank-(N-2); tr(rank-2) -> a rank-0 scalar tensor.
        auto r4 = random::random_tensor({2, 3, 2, 4}, -1.0, 1.0, Device.cpu, 0, Type.Double);
        auto out4 = linalg::Trace(r4, 0, 2);
        EXPECT_EQ(out4.shape().size(), 2u);
        EXPECT_EQ(out4.shape()[0], 3u);
        EXPECT_EQ(out4.shape()[1], 4u);

        auto r3 = random::random_tensor({3, 4, 3}, -1.0, 1.0, Device.cpu, 0, Type.Double);
        auto out3 = linalg::Trace(r3, 0, 2);
        EXPECT_EQ(out3.shape().size(), 1u);
        EXPECT_EQ(out3.shape()[0], 4u);

        auto r2 = random::random_tensor({5, 5}, -1.0, 1.0, Device.cpu, 0, Type.Double);
        auto out2 = linalg::Trace(r2, 0, 1);
        EXPECT_TRUE(out2.is_scalar());
      }

      TEST(LinalgTraceTest, SwappedAxisOrderMatches) {
        // Trace(T, a, b) == Trace(T, b, a) (the function normalizes the order).
        auto t = random::random_tensor({3, 4, 3, 2}, -1.0, 1.0, Device.cpu, 0, Type.Double);
        auto ab = linalg::Trace(t, 0, 2);
        auto ba = linalg::Trace(t, 2, 0);
        EXPECT_TRUE(test::AreNearlyEqTensor(ab, ba, 1e-12));
      }

      // The following three cases match NumPy's convention for zero-extent np.trace
      // (verified against numpy.trace with axis1/axis2 on shapes {0,0}, {3,0,0},
      // {3,0,3}): a zero-length diagonal sums to 0 for every surviving index, and a
      // zero-length surviving axis produces an empty result of that shape.

      TEST(LinalgTraceTest, Rank2ZeroExtentReturnsZeroScalar) {
        auto t = ZeroExtentTensor({0, 0}, Type.Double);
        auto out = linalg::Trace(t, 0, 1);
        ASSERT_TRUE(out.is_scalar());
        EXPECT_DOUBLE_EQ(out.item<cytnx_double>(), 0.0);
      }

      TEST(LinalgTraceTest, ZeroTracedAxisReturnsZeroFilledOutput) {
        // shape {3, 0, 0} traced over (1, 2): the diagonal is empty for every one of
        // the 3 surviving indices, so each sums to 0 -> output shape {3}, all zero.
        auto t = ZeroExtentTensor({3, 0, 0}, Type.Double);
        auto out = linalg::Trace(t, 1, 2);
        ASSERT_EQ(out.shape().size(), 1u);
        EXPECT_EQ(out.shape()[0], 3u);
        for (cytnx_uint64 i = 0; i < 3; i++) EXPECT_DOUBLE_EQ(out.at<cytnx_double>({i}), 0.0);
      }

      TEST(LinalgTraceTest, ZeroRemainingAxisReturnsEmptyOutput) {
        // shape {3, 0, 3} traced over (0, 2): the surviving axis (1) is zero-length,
        // so the output is an empty tensor of shape {0}.
        auto t = ZeroExtentTensor({3, 0, 3}, Type.Double);
        auto out = linalg::Trace(t, 0, 2);
        ASSERT_EQ(out.shape().size(), 1u);
        EXPECT_EQ(out.shape()[0], 0u);
      }

    }  // namespace
  }  // namespace test

}  // namespace cytnx
