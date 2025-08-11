#include "Storage_test.h"

#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "test_tools.h"

TEST_F(StorageTest, dtype_str) {
  std::vector<cytnx_complex128> vcd = {cytnx_complex128(1, 2), cytnx_complex128(3, 4),
                                       cytnx_complex128(5, 6)};

  Storage sd = Storage::from_vector(vcd);
  EXPECT_EQ(sd.dtype_str(), Type.getname(Type.ComplexDouble));
}

TEST_F(StorageTest, device_str) {
  std::vector<cytnx_complex128> vcd = {cytnx_complex128(1, 2), cytnx_complex128(3, 4),
                                       cytnx_complex128(5, 6)};

  Storage sd = Storage::from_vector(vcd);
  EXPECT_EQ(sd.device_str(), Device.getname(Device.cpu));
}

TEST_F(StorageTest, Get_real_cd) {
  std::vector<cytnx_complex128> vcd = {cytnx_complex128(1, 2), cytnx_complex128(3, 4),
                                       cytnx_complex128(5, 6)};

  Storage sd = Storage::from_vector(vcd);

  Storage real_sd = sd.real();

  EXPECT_EQ(real_sd.at<double>(0), double(1));
  EXPECT_EQ(real_sd.at<double>(1), double(3));
  EXPECT_EQ(real_sd.at<double>(2), double(5));
}

TEST_F(StorageTest, Get_imag_cd) {
  std::vector<cytnx_complex128> vcd = {cytnx_complex128(1, 2), cytnx_complex128(3, 4),
                                       cytnx_complex128(5, 6)};

  Storage sd = Storage::from_vector(vcd);

  Storage im_sd = sd.imag();

  EXPECT_EQ(im_sd.at<double>(0), double(2));
  EXPECT_EQ(im_sd.at<double>(1), double(4));
  EXPECT_EQ(im_sd.at<double>(2), double(6));
}

TEST_F(StorageTest, Get_real_cf) {
  std::vector<cytnx_complex64> vcf = {cytnx_complex64(1, 2), cytnx_complex64(3, 4),
                                      cytnx_complex64(5, 6)};

  Storage sd = Storage::from_vector(vcf);

  Storage real_sd = sd.real();

  EXPECT_EQ(real_sd.at<float>(0), float(1));
  EXPECT_EQ(real_sd.at<float>(1), float(3));
  EXPECT_EQ(real_sd.at<float>(2), float(5));
}

TEST_F(StorageTest, Get_imag_cf) {
  std::vector<cytnx_complex64> vcf = {cytnx_complex64(1, 2), cytnx_complex64(3, 4),
                                      cytnx_complex64(5, 6)};

  Storage sd = Storage::from_vector(vcf);

  Storage im_sd = sd.imag();

  EXPECT_EQ(im_sd.at<float>(0), float(2));
  EXPECT_EQ(im_sd.at<float>(1), float(4));
  EXPECT_EQ(im_sd.at<float>(2), float(6));
}

// test fromvector:

TEST_F(StorageTest, from_vec_cd) {
  auto e1 = cytnx_complex128(1, 2);
  auto e2 = cytnx_complex128(3, 4);
  std::vector<cytnx_complex128> vcd = {e1, e2};
  Storage sd = Storage::from_vector(vcd);

  EXPECT_EQ(sd.dtype(), Type.ComplexDouble);
  EXPECT_EQ(sd.at<cytnx_complex128>(0), e1);
  EXPECT_EQ(sd.at<cytnx_complex128>(1), e2);
}

TEST_F(StorageTest, from_vec_cf) {
  auto e1 = cytnx_complex64(1, 2);
  auto e2 = cytnx_complex64(3, 4);
  std::vector<cytnx_complex64> vcd = {e1, e2};
  Storage sd = Storage::from_vector(vcd);

  EXPECT_EQ(sd.dtype(), Type.ComplexFloat);
  EXPECT_EQ(sd.at<cytnx_complex64>(0), e1);
  EXPECT_EQ(sd.at<cytnx_complex64>(1), e2);
}

// create suite for all real types (exclude bool)
using vector_typelist = testing::Types<cytnx_int64, cytnx_uint64, cytnx_int32, cytnx_uint32,
                                       cytnx_double, cytnx_float, cytnx_uint16, cytnx_int16>;
template <class>
struct vector_suite : testing::Test {};
TYPED_TEST_SUITE(vector_suite, vector_typelist);

// generic testing for all type:
TYPED_TEST(vector_suite, from_vec_real) {
  auto e1 = TypeParam(2);
  auto e2 = TypeParam(7);

  std::vector<TypeParam> v = {e1, e2};
  Storage sd = Storage::from_vector(v);

  EXPECT_EQ(sd.dtype(), Type.cy_typeid(e1));
  EXPECT_EQ(sd.at<TypeParam>(0), e1);
  EXPECT_EQ(sd.at<TypeParam>(1), e2);
}

TYPED_TEST(vector_suite, storage_cpu_to_cpu) {
  auto e1 = TypeParam(2);
  auto e2 = TypeParam(7);

  std::vector<TypeParam> v = {e1, e2};
  Storage sd = Storage::from_vector(v);

  EXPECT_EQ(sd.dtype(), Type.cy_typeid(e1));

  Storage tar_sd = sd.to(Device.cpu);
  EXPECT_EQ(tar_sd.device(), Device.cpu);
  EXPECT_EQ(is(tar_sd, sd), true);
}

using TestingValueDTypes =
  std::tuple<Scalar, cytnx_complex128, cytnx_complex64, cytnx_double, cytnx_float, cytnx_uint64,
             cytnx_int64, cytnx_uint32, cytnx_int32, cytnx_uint16, cytnx_int16, cytnx_bool>;

using TestingStorageDTypes =
  std::tuple<cytnx_complex128, cytnx_complex64, cytnx_double, cytnx_float, cytnx_uint64,
             cytnx_int64, cytnx_uint32, cytnx_int32, cytnx_uint16, cytnx_int16, cytnx_bool>;

using TestingPutValueTypes =
  TestTools::TestTypeCombinations<TestingStorageDTypes, TestingValueDTypes>;

template <class TypeParam>
struct StoragePutValue : testing::Test {
  using StorageDType = typename TypeParam::first_type;
  using ValueDType = typename TypeParam::second_type;

  /**
   * Try to convert the given value to the data type of the storage.
   *
   * @return `std::false_type{}` if the given value cannot be converted to the data type of the
   * storage, otherwise, return the converted value
   */
  auto TryConvertToStorageDType(const ValueDType& value) {
    if constexpr (is_complex_v<ValueDType> && is_complex_v<StorageDType>) {
      using ValueDTypeInComplex = typename ValueDType::value_type;
      using StorageDTypeInComplex = typename StorageDType::value_type;
      if constexpr (is_convertible_v<ValueDTypeInComplex, StorageDTypeInComplex>) {
        return StorageDType{StorageDTypeInComplex(value.real()),
                            StorageDTypeInComplex(value.imag())};
      } else {
        return std::false_type{};
      }
    } else if constexpr (std::is_same_v<ValueDType, Scalar> &&
                         std::is_same_v<StorageDType, cytnx_complex128>) {
      // There is no conversion function for converting a value from Scalar to complex types.
      return complex128(value);
    } else if constexpr (std::is_same_v<ValueDType, Scalar> &&
                         std::is_same_v<StorageDType, cytnx_complex64>) {
      return complex64(value);
    } else if constexpr (std::is_constructible_v<StorageDType, ValueDType>) {
      // handle both implicit conversion and explicit conversion
      return StorageDType(value);
    } else {
      return std::false_type{};
    }
  }
};
TYPED_TEST_SUITE(StoragePutValue, TestingPutValueTypes);

TYPED_TEST(StoragePutValue, Fill) {
  using StorageDType = typename TestFixture::StorageDType;
  using ValueDType = typename TestFixture::ValueDType;
  if constexpr (std::is_same_v<ValueDType, Scalar>) {
    GTEST_SKIP() << "Filling a storage with a Scalar is not supported.";
  } else {
    auto element1 = StorageDType(2);
    auto element2 = StorageDType(7);

    std::vector<StorageDType> v = {element1, element2};
    Storage storage = Storage::from_vector(v);

    auto value_to_fill = ValueDType(10);
    auto value_in_storage_dtype = this->TryConvertToStorageDType(value_to_fill);
    if constexpr (std::is_same_v<decltype(value_in_storage_dtype), std::false_type>) {
      EXPECT_THROW(storage.fill(value_to_fill), std::logic_error);
    } else {
      storage.fill(value_to_fill);
      EXPECT_EQ(storage.at<StorageDType>(0), StorageDType(value_to_fill));
      EXPECT_EQ(storage.at<StorageDType>(1), StorageDType(value_to_fill));
    }
  }
}

TYPED_TEST(StoragePutValue, AppendWithReallocation) {
  using StorageDType = typename TestFixture::StorageDType;
  using ValueDType = typename TestFixture::ValueDType;

  auto element1 = StorageDType(2);
  auto element2 = StorageDType(7);

  std::vector<StorageDType> v = {element1, element2};
  Storage storage = Storage::from_vector(v);

  ASSERT_EQ(storage.dtype(), Type.cy_typeid(element1));
  ASSERT_EQ(storage.size(), storage.capacity());

  auto value_to_append = ValueDType(10);
  auto value_in_storage_dtype = this->TryConvertToStorageDType(value_to_append);
  constexpr bool is_not_convertible =
    std::is_same_v<decltype(value_in_storage_dtype), std::false_type>;
  if constexpr (is_not_convertible) {
    EXPECT_THROW(storage.append(value_to_append), std::logic_error);
  } else {
    storage.append(value_to_append);
    EXPECT_GE(storage.capacity(), storage.size());
    EXPECT_EQ(storage.size(), 3);
  }
}
