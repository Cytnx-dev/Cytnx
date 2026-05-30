#include <gtest/gtest.h>

#include <functional>
#include <vector>

#include "Tensor.hpp"
#include "Type.hpp"
#include "random.hpp"

namespace {

  // For a tensor T (possibly permuted / non-contiguous), the storage offset
  // implied by strides() at multi-index idx must equal the offset that at(idx)
  // actually reads -- otherwise the trace's stride-aware diagonal sum would
  // disagree with Tensor::at across permutations.
  template <typename T>
  void ExpectStridesMatchAt(const cytnx::Tensor& tensor) {
    const auto shape = tensor.shape();
    const auto strides = tensor.strides();
    ASSERT_EQ(strides.size(), shape.size());
    std::vector<cytnx::cytnx_uint64> idx(shape.size(), 0);
    std::function<void(std::size_t)> recurse = [&](std::size_t axis) {
      if (axis == shape.size()) {
        cytnx::cytnx_uint64 offset = 0;
        for (std::size_t a = 0; a < shape.size(); ++a) offset += idx[a] * strides[a];
        EXPECT_EQ(tensor.at<T>(idx), tensor.storage().at<T>(offset));
      } else {
        for (idx[axis] = 0; idx[axis] < shape[axis]; ++idx[axis]) recurse(axis + 1);
      }
    };
    recurse(0);
  }

  TEST(TensorStridesTest, ContiguousIsRowMajor) {
    auto tensor =
      cytnx::random::random_tensor({4, 5, 3}, -1.0, 1.0, cytnx::Device.cpu, 0, cytnx::Type.Double);
    ASSERT_TRUE(tensor.is_contiguous());
    const auto s = tensor.strides();
    // Row-major: strides = (5*3, 3, 1).
    ASSERT_EQ(s.size(), 3u);
    EXPECT_EQ(s[0], 15u);
    EXPECT_EQ(s[1], 3u);
    EXPECT_EQ(s[2], 1u);
    ExpectStridesMatchAt<cytnx::cytnx_double>(tensor);
  }

  TEST(TensorStridesTest, PermutedMatchesAtForRanks2to5) {
    using cytnx::cytnx_uint64;
    struct Case {
      std::vector<cytnx_uint64> shape;
      std::vector<cytnx_uint64> perm;
    };
    const std::vector<Case> cases = {
      {{3, 4}, {1, 0}},
      {{3, 4, 5}, {2, 0, 1}},
      {{2, 3, 4, 2}, {3, 1, 0, 2}},
      {{2, 3, 2, 3, 2}, {4, 2, 0, 3, 1}},
    };
    for (const auto& c : cases) {
      auto t =
        cytnx::random::random_tensor(c.shape, -1.0, 1.0, cytnx::Device.cpu, 0, cytnx::Type.Double);
      auto p = t.permute(c.perm);
      EXPECT_FALSE(p.is_contiguous()) << "expected the permutation to be non-contiguous";
      ExpectStridesMatchAt<cytnx::cytnx_double>(p);
    }
  }

  TEST(TensorStridesTest, ComplexAndIntegerDtypes) {
    auto td = cytnx::random::random_tensor({3, 4, 2}, -1.0, 1.0, cytnx::Device.cpu, 0,
                                           cytnx::Type.ComplexDouble);
    ExpectStridesMatchAt<cytnx::cytnx_complex128>(td.permute({2, 0, 1}));

    auto ti =
      cytnx::random::random_tensor({3, 4, 2}, 0.0, 10.0, cytnx::Device.cpu, 0, cytnx::Type.Int32);
    ExpectStridesMatchAt<cytnx::cytnx_int32>(ti.permute({1, 2, 0}));
  }

}  // namespace
