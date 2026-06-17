#include <gtest/gtest.h>

#include "Generator.hpp"
#include "TensorT.hpp"
#include "TensorT_traits.hpp"

#include <array>

namespace {

  using cytnx::arange;
  using cytnx::ComplexTensor;
  using cytnx::cytnx_double;
  using cytnx::Device;
  using cytnx::host_access;
  using cytnx::HostTensorT;
  using cytnx::make_right_tensor_t;
  using cytnx::make_tensor;
  using cytnx::make_tensor_t;
  using cytnx::NumericTensor;
  using cytnx::RealTensor;
  using cytnx::Tensor;
  using cytnx::to_tensor;
  using cytnx::Type;
  using cytnx::stdex::layout_right;
  using cytnx::stdex::layout_stride;

  static_assert(cytnx::RealScalar<cytnx::cytnx_float>);
  static_assert(cytnx::RealScalar<cytnx::cytnx_double>);
  static_assert(!cytnx::RealScalar<cytnx::cytnx_complex64>);
  static_assert(cytnx::ComplexScalar<cytnx::cytnx_complex64>);
  static_assert(cytnx::ComplexScalar<cytnx::cytnx_complex128>);
  static_assert(!cytnx::ComplexScalar<cytnx::cytnx_double>);
  static_assert(cytnx::NumericScalar<cytnx::cytnx_float>);
  static_assert(cytnx::NumericScalar<cytnx::cytnx_complex128>);

#ifdef UNI_GPU
  static_assert(std::variant_size_v<NumericTensor<2>> == 8);
  static_assert(std::variant_size_v<RealTensor<2>> == 4);
#else
  static_assert(std::variant_size_v<NumericTensor<2>> == 4);
  static_assert(std::variant_size_v<RealTensor<2>> == 2);
#endif

  TEST(TensorTTest, DirectAllocationCreatesOwnedContiguousStrideView) {
    HostTensorT<cytnx_double, 2> matrix({2, 3});

    static_assert(std::is_same_v<decltype(matrix), HostTensorT<cytnx_double, 2, layout_stride>>);
    EXPECT_EQ(matrix.dtype(), Type.Double);
    EXPECT_EQ(matrix.device(), Device.cpu);
    EXPECT_EQ(matrix.extent(0), 2);
    EXPECT_EQ(matrix.extent(1), 3);
    EXPECT_EQ(matrix.rows(), 2);
    EXPECT_EQ(matrix.cols(), 3);
    EXPECT_EQ(matrix.stride(0), 3);
    EXPECT_EQ(matrix.stride(1), 1);
    EXPECT_EQ(matrix.required_span_size(), 6);
    EXPECT_TRUE(static_cast<bool>(matrix.owner()));
    EXPECT_EQ(matrix(1, 2), 0);

    matrix(1, 2) = 17;
    EXPECT_EQ(matrix(1, 2), 17);
    EXPECT_THROW(to_tensor(matrix), std::logic_error);
  }

  TEST(TensorTTest, DirectAllocationSupportsRankOneSize) {
    HostTensorT<cytnx_double, 1> vector({5});

    EXPECT_EQ(vector.extent(0), 5);
    EXPECT_EQ(vector.size(), 5);
    EXPECT_EQ(vector.required_span_size(), 5);

    vector(4) = 12;
    EXPECT_EQ(vector(4), 12);
  }

  TEST(TensorTTest, DirectAllocationSupportsLayoutRightAndArrayExtents) {
    HostTensorT<cytnx_double, 3, layout_right> tensor(std::array<std::size_t, 3>{2, 3, 4});

    EXPECT_EQ(tensor.extent(0), 2);
    EXPECT_EQ(tensor.extent(1), 3);
    EXPECT_EQ(tensor.extent(2), 4);
    EXPECT_EQ(tensor.stride(0), 12);
    EXPECT_EQ(tensor.stride(1), 4);
    EXPECT_EQ(tensor.stride(2), 1);
    EXPECT_EQ(tensor.required_span_size(), 24);

    tensor(1, 2, 3) = 29;
    EXPECT_EQ(tensor(1, 2, 3), 29);
  }

  TEST(TensorTTest, DirectAllocationSupportsRankZeroScalars) {
    HostTensorT<cytnx_double, 0> zero(std::array<std::size_t, 0>{});

    static_assert(decltype(zero)::rank() == 0);
    EXPECT_EQ(zero.dtype(), Type.Double);
    EXPECT_EQ(zero.device(), Device.cpu);
    EXPECT_EQ(zero.required_span_size(), 1);
    EXPECT_TRUE(static_cast<bool>(zero.owner()));
    EXPECT_EQ(zero(), 0);

    zero() = 11;
    EXPECT_EQ(zero.value(), 11);

    HostTensorT<cytnx_double, 0> scalar(3.5);
    EXPECT_EQ(scalar(), 3.5);
    scalar.value() = 4.5;
    EXPECT_EQ(scalar(), 4.5);
    EXPECT_THROW(to_tensor(scalar), std::logic_error);
  }

  TEST(TensorTTest, DirectAllocationRejectsWrongNumberOfInitializerExtents) {
    EXPECT_THROW((HostTensorT<cytnx_double, 2>({2, 3, 4})), std::logic_error);
  }

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

  TEST(TensorTTest, GenericVariantFactoryDispatchesDtypeToLayoutRightHostViews) {
    Tensor real = arange(2 * 3).reshape({2, 3});
    auto real_view = make_tensor<RealTensor<2>>(real);
    ASSERT_TRUE((std::holds_alternative<HostTensorT<cytnx_double, 2, layout_right>>(real_view)));

    Tensor complex({2, 3}, Type.ComplexFloat);
    auto complex_view = make_tensor<ComplexTensor<2>>(complex);
    ASSERT_TRUE(
      (std::holds_alternative<HostTensorT<cytnx::cytnx_complex64, 2, layout_right>>(complex_view)));

    auto numeric_view = make_tensor<NumericTensor<2>>(complex);
    ASSERT_TRUE(
      (std::holds_alternative<HostTensorT<cytnx::cytnx_complex64, 2, layout_right>>(numeric_view)));
  }

  TEST(TensorTTest, GenericVariantFactoryWorksWithCustomVariant) {
    using CustomTensor = std::variant<HostTensorT<cytnx_double, 3, layout_stride>,
                                      HostTensorT<cytnx::cytnx_complex128, 3, layout_stride>>;

    Tensor tensor = arange(2 * 3 * 4).reshape({2, 3, 4});
    Tensor permuted = tensor.permute({1, 2, 0});

    auto variant = make_tensor<CustomTensor>(permuted);

    ASSERT_TRUE((std::holds_alternative<HostTensorT<cytnx_double, 3, layout_stride>>(variant)));
    auto &view = std::get<HostTensorT<cytnx_double, 3, layout_stride>>(variant);
    EXPECT_EQ(view.extent(0), 3);
    EXPECT_EQ(view.extent(1), 4);
    EXPECT_EQ(view.extent(2), 2);
    EXPECT_EQ(view.stride(0), 4);
    EXPECT_EQ(view.stride(1), 1);
    EXPECT_EQ(view.stride(2), 12);

    Tensor wrong_rank = arange(2 * 3).reshape({2, 3});
    EXPECT_THROW((make_tensor<CustomTensor>(wrong_rank)), std::logic_error);
  }

  TEST(TensorTTest, GenericVariantFactoryDoesNotMutateConstInputForLayoutRight) {
    Tensor tensor = arange(2 * 3 * 4).reshape({2, 3, 4});
    Tensor permuted = tensor.permute({1, 2, 0});
    const Tensor &input = permuted;

    auto variant = make_tensor<NumericTensor<3>>(input);

    ASSERT_TRUE((std::holds_alternative<HostTensorT<cytnx_double, 3, layout_right>>(variant)));
    EXPECT_FALSE(permuted.is_contiguous());

    auto &view = std::get<HostTensorT<cytnx_double, 3, layout_right>>(variant);
    EXPECT_EQ(view.extent(0), 3);
    EXPECT_EQ(view.extent(1), 4);
    EXPECT_EQ(view.extent(2), 2);
    EXPECT_EQ(view(2, 3, 1), tensor.at<cytnx_double>({1, 2, 3}));

    view(2, 3, 1) = 99;
    EXPECT_EQ(tensor.at<cytnx_double>({1, 2, 3}), 23);
  }

  TEST(TensorTTest, RejectsWrongDtypeAndRank) {
    Tensor tensor = arange(2 * 3).reshape({2, 3});

    EXPECT_THROW((make_tensor_t<float, 2>(tensor)), std::logic_error);
    EXPECT_THROW((make_tensor_t<cytnx_double, 3>(tensor)), std::logic_error);
  }

}  // namespace
