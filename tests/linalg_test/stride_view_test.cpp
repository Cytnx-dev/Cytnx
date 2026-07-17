#include <complex>
#include <ranges>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "backend/linalg_internal_cpu/pairwise_sum.hpp"
#include "backend/linalg_internal_cpu/stride_view.hpp"
namespace cytnx {
  namespace {

    using linalg_internal::PairwiseSum;
    using linalg_internal::stride;
    using linalg_internal::stride_view;

    using DoubleView = stride_view<std::span<const double>>;

    // stride_view must model a sized random-access range so PairwiseSum (and any
    // other range algorithm) can consume it.
    static_assert(std::random_access_iterator<DoubleView::iterator>);
    static_assert(std::ranges::random_access_range<DoubleView>);
    static_assert(std::ranges::sized_range<DoubleView>);

    TEST(StrideViewTest, SelectsEveryStepthElement) {
      std::vector<double> v(12);
      for (int i = 0; i < 12; ++i) v[i] = i;
      auto sv = stride_view(std::span<const double>(v), 3);
      ASSERT_EQ(sv.size(), 4u);
      std::vector<double> got(sv.begin(), sv.end());
      ASSERT_EQ(got.size(), 4u);
      for (int k = 0; k < 4; ++k) EXPECT_DOUBLE_EQ(got[k], 3.0 * k);
    }

    TEST(StrideViewTest, SizeRoundsUpWhenLengthNotMultipleOfStride) {
      // When the range length is not an exact multiple of the stride, size() must
      // round up: a 10-element range with stride 3 yields indices 0, 3, 6, 9. This
      // exercises the (n + step - 1) / step branch in size() and end().
      std::vector<double> v(10);
      for (int i = 0; i < 10; ++i) v[i] = i;
      auto sv = stride_view(std::span<const double>(v), 3);
      ASSERT_EQ(sv.size(), 4u);
      EXPECT_EQ(sv.end() - sv.begin(), 4);
      std::vector<double> got(sv.begin(), sv.end());
      ASSERT_EQ(got.size(), 4u);
      EXPECT_DOUBLE_EQ(got[0], 0.0);
      EXPECT_DOUBLE_EQ(got[1], 3.0);
      EXPECT_DOUBLE_EQ(got[2], 6.0);
      EXPECT_DOUBLE_EQ(got[3], 9.0);
    }

    TEST(StrideViewTest, RejectsZeroStride) {
      std::vector<double> v(4);
      EXPECT_THROW(stride_view(std::span<const double>(v), 0), std::invalid_argument);
    }

    TEST(StrideViewTest, EmptyBaseYieldsEmptyView) {
      // An empty underlying range must yield size()==0 and begin()==end(); a
      // sum/reduce algorithm consuming it must produce the identity.
      std::vector<double> empty;
      auto sv = stride_view(std::span<const double>(empty), 4);
      EXPECT_EQ(sv.size(), 0u);
      EXPECT_EQ(sv.begin(), sv.end());
      double s = PairwiseSum(std::span<const double>(empty) | stride(4));
      EXPECT_DOUBLE_EQ(s, 0.0);
    }

    TEST(StrideViewTest, StrideLargerThanBaseSelectsFirstElement) {
      // When step exceeds the underlying length, the view selects exactly one
      // element (the first). This is the (n + step - 1) / step formula's
      // single-element regime and keeps the view well-formed for short inputs.
      std::vector<double> v = {7.0, 8.0, 9.0};
      auto sv = stride_view(std::span<const double>(v), 10);
      EXPECT_EQ(sv.size(), 1u);
      std::vector<double> got(sv.begin(), sv.end());
      ASSERT_EQ(got.size(), 1u);
      EXPECT_DOUBLE_EQ(got[0], 7.0);
    }

    TEST(StrideViewTest, PipeAdaptorFeedsPairwiseSum) {
      std::vector<double> v(30);
      for (int i = 0; i < 30; ++i) v[i] = i;
      // 0 + 3 + 6 + ... + 27 = 135
      double s = PairwiseSum(std::span<const double>(v) | stride(3));
      EXPECT_DOUBLE_EQ(s, 135.0);
    }

    TEST(StrideViewTest, IteratorsOutliveTheView) {
      // The iterator stores the underlying base iterator, not a pointer to the
      // stride_view, so iterators stay valid after the view is destroyed as long
      // as the underlying data lives.
      std::vector<double> w(10, 2.0);
      DoubleView::iterator it;
      DoubleView::iterator end;
      {
        auto sv = stride_view(std::span<const double>(w), 2);
        it = sv.begin();
        end = sv.end();
      }
      double s = 0;
      for (; it != end; ++it) s += *it;
      EXPECT_DOUBLE_EQ(s, 10.0);
    }

    TEST(StrideViewTest, RandomAccessOps) {
      std::vector<double> v(20);
      for (int i = 0; i < 20; ++i) v[i] = i;
      auto sv = stride_view(std::span<const double>(v), 2);
      auto it = sv.begin();
      EXPECT_DOUBLE_EQ(*(it + 3), 6.0);
      EXPECT_DOUBLE_EQ(it[4], 8.0);
      // Negative offsets must step backwards: step_ is the signed difference_type,
      // so n * step_ stays signed and a negative index does not wrap around.
      auto mid = sv.begin() + 5;
      EXPECT_DOUBLE_EQ(mid[-2], 6.0);
      EXPECT_DOUBLE_EQ(*(sv.end() - 1), 18.0);
      EXPECT_EQ(sv.end() - sv.begin(), 10);
      EXPECT_TRUE(sv.begin() < sv.end());
    }

    TEST(StrideViewTest, ComplexStridedSum) {
      std::vector<std::complex<double>> v(12, std::complex<double>(1.0, -1.0));
      auto s = PairwiseSum(std::span<const std::complex<double>>(v) | stride(2));  // 6 elements
      EXPECT_DOUBLE_EQ(s.real(), 6.0);
      EXPECT_DOUBLE_EQ(s.imag(), -6.0);
    }

  }  // namespace

}  // namespace cytnx
