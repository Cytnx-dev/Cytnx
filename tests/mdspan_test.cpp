#include <gtest/gtest.h>

#include "mdspan.hpp"

#include <array>
#include <vector>

namespace {

  using cytnx::stdex::dynamic_extent;
  using cytnx::stdex::extents;
  using cytnx::stdex::layout_right;
  using cytnx::stdex::layout_stride;
  using cytnx::stdex::mdspan;

  TEST(MdspanTest, StaticExtentsLayoutRightIndexing) {
    std::array<int, 6> data{0, 1, 2, 3, 4, 5};
    mdspan<int, extents<std::size_t, 2, 3>> view(data.data());

    EXPECT_EQ(view.rank(), 2);
    EXPECT_EQ(view.extent(0), 2);
    EXPECT_EQ(view.extent(1), 3);
    EXPECT_EQ(view.stride(0), 3);
    EXPECT_EQ(view.stride(1), 1);
    EXPECT_EQ(view.required_span_size(), 6);
    EXPECT_EQ(view(0, 0), 0);
    EXPECT_EQ(view(0, 2), 2);
    EXPECT_EQ(view(1, 0), 3);
    EXPECT_EQ(view(1, 2), 5);
  }

  TEST(MdspanTest, DynamicExtentsKeepCompileTimeRank) {
    std::vector<double> data(12);
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = static_cast<double>(i);

    mdspan<double, extents<std::size_t, dynamic_extent, 4>> view(data.data(), 3);

    EXPECT_EQ(view.rank(), 2);
    EXPECT_EQ(view.rank_dynamic(), 1);
    EXPECT_EQ(view.extent(0), 3);
    EXPECT_EQ(view.extent(1), 4);
    EXPECT_EQ(view.stride(0), 4);
    EXPECT_EQ(view.stride(1), 1);
    EXPECT_EQ(view(2, 3), 11.0);
  }

  TEST(MdspanTest, LayoutStrideIndexing) {
    std::array<int, 6> data{0, 1, 2, 3, 4, 5};
    using extents_type = extents<std::size_t, 3, 2>;
    using mapping_type = layout_stride::mapping<extents_type>;

    mapping_type mapping(extents_type(), std::array<std::size_t, 2>{1, 3});
    mdspan<int, extents_type, layout_stride> view(data.data(), mapping);

    EXPECT_EQ(view.rank(), 2);
    EXPECT_EQ(view.extent(0), 3);
    EXPECT_EQ(view.extent(1), 2);
    EXPECT_EQ(view.stride(0), 1);
    EXPECT_EQ(view.stride(1), 3);
    EXPECT_EQ(view.required_span_size(), 6);
    EXPECT_EQ(view(0, 0), 0);
    EXPECT_EQ(view(2, 0), 2);
    EXPECT_EQ(view(0, 1), 3);
    EXPECT_EQ(view(2, 1), 5);
  }

  TEST(MdspanTest, ElementAccessMutatesUnderlyingData) {
    std::array<int, 4> data{0, 1, 2, 3};
    mdspan<int, extents<std::size_t, 2, 2>> view(data.data());

    view(1, 0) = 42;

    EXPECT_EQ(data[2], 42);
  }

}  // namespace
