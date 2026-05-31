#include <gtest/gtest.h>

#include <complex>
#include <ranges>
#include <span>
#include <vector>

#include "backend/linalg_internal_cpu/pairwise_sum.hpp"
#include "backend/linalg_internal_cpu/stride_view.hpp"

namespace {

  using cytnx::linalg_internal::PairwiseSum;
  using cytnx::linalg_internal::stride;
  using cytnx::linalg_internal::stride_view;

  using DoubleView = stride_view<std::span<const double>>;

  // stride_view must model a sized random-access range so PairwiseSum (and any
  // other range algorithm) can consume it.
  static_assert(std::random_access_iterator<DoubleView::iterator>);
  static_assert(std::ranges::random_access_range<DoubleView>);
  static_assert(std::ranges::sized_range<DoubleView>);

  TEST(StrideViewTest, SelectsEveryStepThElement) {
    std::vector<double> v(30);
    for (int i = 0; i < 30; ++i) v[i] = i;
    stride_view sv(std::span<const double>(v), 3);
    ASSERT_EQ(sv.size(), 10u);
    std::vector<double> got(sv.begin(), sv.end());
    ASSERT_EQ(got.size(), 10u);
    for (int k = 0; k < 10; ++k) EXPECT_DOUBLE_EQ(got[k], 3.0 * k);
  }

  TEST(StrideViewTest, NonDividingRangeRoundsUp) {
    // size 10 with stride 3 -> 4 elements at offsets 0, 3, 6, 9 (exercises the
    // n % step != 0 branch of size(), which is the shape produced by Trace's
    // extent = (Ndiag - 1) * step + 1 contract).
    std::vector<double> v(10);
    for (int i = 0; i < 10; ++i) v[i] = i;
    stride_view sv(std::span<const double>(v), 3);
    ASSERT_EQ(sv.size(), 4u);
    std::vector<double> got(sv.begin(), sv.end());
    ASSERT_EQ(got.size(), 4u);
    for (int k = 0; k < 4; ++k) EXPECT_DOUBLE_EQ(got[k], 3.0 * k);
  }

  TEST(StrideViewTest, RejectsZeroStride) {
    std::vector<double> v(4);
    EXPECT_THROW(stride_view(std::span<const double>(v), 0), std::invalid_argument);
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
    DoubleView::iterator b, e;
    {
      stride_view sv(std::span<const double>(w), 2);
      b = sv.begin();
      e = sv.end();
    }
    double s = 0;
    for (auto it = b; it != e; ++it) s += *it;
    EXPECT_DOUBLE_EQ(s, 10.0);  // 5 selected elements * 2.0
  }

  TEST(StrideViewTest, RandomAccessOps) {
    std::vector<double> v(20);
    for (int i = 0; i < 20; ++i) v[i] = i;
    stride_view sv(std::span<const double>(v), 2);  // 0,2,4,...,18
    auto it = sv.begin();
    EXPECT_DOUBLE_EQ(*(it + 3), 6.0);
    EXPECT_DOUBLE_EQ(it[4], 8.0);
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
