#include <gtest/gtest.h>

#include "Generator.hpp"
#include "TensorT.hpp"

namespace {

  using cytnx::arange;
  using cytnx::cytnx_double;
  using cytnx::Device;
  using cytnx::host_access;
  using cytnx::HostTensorT;
  using cytnx::make_right_tensor_t;
  using cytnx::make_tensor_t;
  using cytnx::Tensor;
  using cytnx::to_tensor;
  using cytnx::Type;
  using cytnx::stdex::layout_right;
  using cytnx::stdex::layout_stride;

  TEST(TensorTTest, MakeTensorTPreservesPermutedStrides) {
    Tensor tensor = arange(2 * 3 * 4).reshape({2, 3, 4});
    Tensor permuted = tensor.permute({1, 2, 0});

    auto view = make_tensor_t<cytnx_double, 3, host_access>(permuted);

    static_assert(std::is_same_v<decltype(view), HostTensorT<cytnx_double, 3, layout_stride>>);
    EXPECT_EQ(view.dtype(), Type.Double);
    EXPECT_EQ(view.device(), Device.cpu);
    EXPECT_EQ(view.extent(0), 3);
    EXPECT_EQ(view.extent(1), 4);
    EXPECT_EQ(view.extent(2), 2);
    EXPECT_EQ(view.stride(0), 4);
    EXPECT_EQ(view.stride(1), 1);
    EXPECT_EQ(view.stride(2), 12);
    EXPECT_EQ(view(1, 2, 0), tensor.at<cytnx_double>({0, 1, 2}));

    view(2, 3, 1) = 123;
    EXPECT_EQ(tensor.at<cytnx_double>({1, 2, 3}), 123);
  }

  TEST(TensorTTest, MakeRightTensorTMutatesToContiguousLayoutRight) {
    Tensor tensor = arange(2 * 3 * 4).reshape({2, 3, 4});
    Tensor permuted = tensor.permute({1, 2, 0});

    auto view = make_right_tensor_t<cytnx_double, 3, host_access>(permuted);

    static_assert(std::is_same_v<decltype(view), HostTensorT<cytnx_double, 3, layout_right>>);
    EXPECT_TRUE(permuted.is_contiguous());
    EXPECT_EQ(view.extent(0), 3);
    EXPECT_EQ(view.extent(1), 4);
    EXPECT_EQ(view.extent(2), 2);
    EXPECT_EQ(view.stride(0), 8);
    EXPECT_EQ(view.stride(1), 2);
    EXPECT_EQ(view.stride(2), 1);
    EXPECT_EQ(view(2, 3, 1), tensor.at<cytnx_double>({1, 2, 3}));

    view(2, 3, 1) = 321;
    EXPECT_EQ(permuted.at<cytnx_double>({2, 3, 1}), 321);
    EXPECT_EQ(tensor.at<cytnx_double>({1, 2, 3}), 23);
  }

  TEST(TensorTTest, StorageOwnerKeepsViewAliveAfterTensorIsDestroyed) {
    HostTensorT<cytnx_double, 2, layout_stride> view;
    {
      Tensor tensor = arange(2 * 3).reshape({2, 3});
      view = make_tensor_t<cytnx_double, 2>(tensor);
    }

    EXPECT_EQ(view.extent(0), 2);
    EXPECT_EQ(view.extent(1), 3);
    EXPECT_EQ(view(1, 2), 5);

    view(1, 2) = 77;
    EXPECT_EQ(view(1, 2), 77);
  }

  TEST(TensorTTest, LegacyStorageCanBeRecoveredFromOwnerDeleter) {
    Tensor tensor = arange(2 * 3).reshape({2, 3});

    auto view = make_tensor_t<cytnx_double, 2>(tensor);
    auto *deleter = std::get_deleter<cytnx::tensor_t_detail::legacy_storage_deleter<cytnx_double>>(
      view.owner().shared_ptr());

    ASSERT_NE(deleter, nullptr);
    ASSERT_TRUE(static_cast<bool>(deleter->storage()));
    EXPECT_EQ(deleter->storage().get(), tensor._impl->storage()._impl.get());
    EXPECT_EQ(deleter->storage()->dtype(), Type.Double);
    EXPECT_EQ(deleter->storage()->device(), Device.cpu);
  }

  TEST(TensorTTest, ToTensorSharesContiguousStorage) {
    Tensor tensor = arange(2 * 3).reshape({2, 3});
    auto view = make_tensor_t<cytnx_double, 2>(tensor);

    Tensor round_trip = to_tensor(view);

    EXPECT_EQ(round_trip.shape(), tensor.shape());
    EXPECT_TRUE(round_trip.is_contiguous());
    EXPECT_EQ(round_trip.at<cytnx_double>({1, 2}), 5);

    round_trip.at<cytnx_double>({1, 2}) = 42;
    EXPECT_EQ(tensor.at<cytnx_double>({1, 2}), 42);
  }

  TEST(TensorTTest, ToTensorSharesPermutedStorage) {
    Tensor tensor = arange(2 * 3 * 4).reshape({2, 3, 4});
    Tensor permuted = tensor.permute({1, 2, 0});
    auto view = make_tensor_t<cytnx_double, 3>(permuted);

    Tensor round_trip = to_tensor(view);

    EXPECT_EQ(round_trip.shape(), permuted.shape());
    EXPECT_FALSE(round_trip.is_contiguous());
    EXPECT_EQ(round_trip.at<cytnx_double>({2, 3, 1}), tensor.at<cytnx_double>({1, 2, 3}));

    round_trip.at<cytnx_double>({2, 3, 1}) = 55;
    EXPECT_EQ(tensor.at<cytnx_double>({1, 2, 3}), 55);
  }

  TEST(TensorTTest, ToTensorRejectsNonPermutationStrides) {
    Tensor tensor = arange(10);
    auto base = make_tensor_t<cytnx_double, 1>(tensor);

    using extents_type = cytnx::stdex::dextents<std::size_t, 2>;
    using mapping_type = layout_stride::mapping<extents_type>;
    using view_type = cytnx::stdex::mdspan<cytnx_double, extents_type, layout_stride>;

    view_type gapped_view(base.data(), mapping_type(extents_type(2, 3), {4, 1}));
    HostTensorT<cytnx_double, 2, layout_stride> gapped(base.owner(), gapped_view);

    EXPECT_THROW(to_tensor(gapped), std::logic_error);
  }

  TEST(TensorTTest, RejectsWrongDtypeAndRank) {
    Tensor tensor = arange(2 * 3).reshape({2, 3});

    EXPECT_THROW((make_tensor_t<float, 2>(tensor)), std::logic_error);
    EXPECT_THROW((make_tensor_t<cytnx_double, 3>(tensor)), std::logic_error);
  }

}  // namespace
