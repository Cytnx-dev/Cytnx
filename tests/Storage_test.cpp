#include "Storage_test.h"

#include <cstdio>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "test_tools.h"

namespace cytnx {
  namespace test {

    class RemoveFileOnExit {
     public:
      explicit RemoveFileOnExit(std::string path) : path_(std::move(path)) {}
      ~RemoveFileOnExit() { std::remove(path_.c_str()); }

      RemoveFileOnExit(const RemoveFileOnExit&) = delete;
      RemoveFileOnExit& operator=(const RemoveFileOnExit&) = delete;

     private:
      std::string path_;
    };

    TEST_F(StorageTest, DtypeStr) {
      std::vector<cytnx_complex128> vcd = {cytnx_complex128(1, 2), cytnx_complex128(3, 4),
                                           cytnx_complex128(5, 6)};

      Storage sd = Storage::from_vector(vcd);
      EXPECT_EQ(sd.dtype_str(), Type.getname(Type.ComplexDouble));
    }

    TEST_F(StorageTest, DeviceStr) {
      std::vector<cytnx_complex128> vcd = {cytnx_complex128(1, 2), cytnx_complex128(3, 4),
                                           cytnx_complex128(5, 6)};

      Storage sd = Storage::from_vector(vcd);
      EXPECT_EQ(sd.device_str(), Device.getname(Device.cpu));
    }

    TEST_F(StorageTest, GetRealCd) {
      std::vector<cytnx_complex128> vcd = {cytnx_complex128(1, 2), cytnx_complex128(3, 4),
                                           cytnx_complex128(5, 6)};

      Storage sd = Storage::from_vector(vcd);

      Storage real_sd = sd.real();

      EXPECT_EQ(real_sd.at<double>(0), double(1));
      EXPECT_EQ(real_sd.at<double>(1), double(3));
      EXPECT_EQ(real_sd.at<double>(2), double(5));
    }

    TEST_F(StorageTest, GetImagCd) {
      std::vector<cytnx_complex128> vcd = {cytnx_complex128(1, 2), cytnx_complex128(3, 4),
                                           cytnx_complex128(5, 6)};

      Storage sd = Storage::from_vector(vcd);

      Storage im_sd = sd.imag();

      EXPECT_EQ(im_sd.at<double>(0), double(2));
      EXPECT_EQ(im_sd.at<double>(1), double(4));
      EXPECT_EQ(im_sd.at<double>(2), double(6));
    }

    TEST_F(StorageTest, GetRealCf) {
      std::vector<cytnx_complex64> vcf = {cytnx_complex64(1, 2), cytnx_complex64(3, 4),
                                          cytnx_complex64(5, 6)};

      Storage sd = Storage::from_vector(vcf);

      Storage real_sd = sd.real();

      EXPECT_EQ(real_sd.at<float>(0), float(1));
      EXPECT_EQ(real_sd.at<float>(1), float(3));
      EXPECT_EQ(real_sd.at<float>(2), float(5));
    }

    TEST_F(StorageTest, GetImagCf) {
      std::vector<cytnx_complex64> vcf = {cytnx_complex64(1, 2), cytnx_complex64(3, 4),
                                          cytnx_complex64(5, 6)};

      Storage sd = Storage::from_vector(vcf);

      Storage im_sd = sd.imag();

      EXPECT_EQ(im_sd.at<float>(0), float(2));
      EXPECT_EQ(im_sd.at<float>(1), float(4));
      EXPECT_EQ(im_sd.at<float>(2), float(6));
    }

    // test fromvector:

    TEST_F(StorageTest, FromVecCd) {
      auto e1 = cytnx_complex128(1, 2);
      auto e2 = cytnx_complex128(3, 4);
      std::vector<cytnx_complex128> vcd = {e1, e2};
      Storage sd = Storage::from_vector(vcd);

      EXPECT_EQ(sd.dtype(), Type.ComplexDouble);
      EXPECT_EQ(sd.at<cytnx_complex128>(0), e1);
      EXPECT_EQ(sd.at<cytnx_complex128>(1), e2);
    }

    TEST_F(StorageTest, FromVecCf) {
      auto e1 = cytnx_complex64(1, 2);
      auto e2 = cytnx_complex64(3, 4);
      std::vector<cytnx_complex64> vcd = {e1, e2};
      Storage sd = Storage::from_vector(vcd);

      EXPECT_EQ(sd.dtype(), Type.ComplexFloat);
      EXPECT_EQ(sd.at<cytnx_complex64>(0), e1);
      EXPECT_EQ(sd.at<cytnx_complex64>(1), e2);
    }

    // create suite for all real types (exclude bool)
    using vector_typelist = ::testing::Types<cytnx_int64, cytnx_uint64, cytnx_int32, cytnx_uint32,
                                             cytnx_double, cytnx_float, cytnx_uint16, cytnx_int16>;
    template <class>
    struct vector_suite : testing::Test {};
    TYPED_TEST_SUITE(vector_suite, vector_typelist);

    // generic testing for all type:
    TYPED_TEST(vector_suite, FromVecReal) {
      auto e1 = TypeParam(2);
      auto e2 = TypeParam(7);

      std::vector<TypeParam> v = {e1, e2};
      Storage sd = Storage::from_vector(v);

      EXPECT_EQ(sd.dtype(), Type.cy_typeid(e1));
      EXPECT_EQ(sd.at<TypeParam>(0), e1);
      EXPECT_EQ(sd.at<TypeParam>(1), e2);
    }

    TYPED_TEST(vector_suite, StorageCpuToCpu) {
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
      test::TestTypeCombinations<TestingStorageDTypes, TestingValueDTypes>;

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
          if constexpr (std::is_convertible_v<ValueDTypeInComplex, StorageDTypeInComplex>) {
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

      auto value_to_append = ValueDType(10);
      auto value_in_storage_dtype = this->TryConvertToStorageDType(value_to_append);
      constexpr bool is_not_convertible =
        std::is_same_v<decltype(value_in_storage_dtype), std::false_type>;
      if constexpr (is_not_convertible) {
        EXPECT_THROW(storage.append(value_to_append), std::logic_error);
      } else {
        storage.append(value_to_append);
        // capacity() was removed (#941 ruling 2): every append reallocates
        // exactly, so the only observable contract is size growth plus element
        // preservation.
        EXPECT_EQ(storage.size(), 3);
        EXPECT_EQ(storage.at<StorageDType>(0), element1);
        EXPECT_EQ(storage.at<StorageDType>(1), element2);
      }
    }

    TEST_F(StorageTest, FromfileHonorsCount) {
      const std::string path = ::testing::TempDir() + "cytnx_storage_fromfile_count.bin";
      std::remove(path.c_str());
      const RemoveFileOnExit cleanup(path);

      Storage source = Storage::from_vector(std::vector<cytnx_double>{1.0, 2.0, 3.0, 4.0});
      source.Tofile(path);

      Storage loaded = Storage::Fromfile(path, Type.Double, 2);
      EXPECT_EQ(loaded.dtype(), Type.Double);
      ASSERT_EQ(loaded.size(), 2);
      EXPECT_DOUBLE_EQ(loaded.at<cytnx_double>(0), 1.0);
      EXPECT_DOUBLE_EQ(loaded.at<cytnx_double>(1), 2.0);
    }

    TEST_F(StorageTest, FromfileRejectsCountLargerThanFile) {
      const std::string path = ::testing::TempDir() + "cytnx_storage_fromfile_oversized_count.bin";
      std::remove(path.c_str());
      const RemoveFileOnExit cleanup(path);

      Storage source = Storage::from_vector(std::vector<cytnx_double>{1.0, 2.0, 3.0, 4.0});
      source.Tofile(path);

      EXPECT_THROW(Storage::Fromfile(path, Type.Double, 5), std::logic_error);
    }

    TEST_F(StorageTest, FromfileCountZeroReturnsEmptyStorage) {
      const std::string path = ::testing::TempDir() + "cytnx_storage_fromfile_count_zero.bin";
      std::remove(path.c_str());
      const RemoveFileOnExit cleanup(path);

      Storage source = Storage::from_vector(std::vector<cytnx_double>{1.0, 2.0, 3.0, 4.0});
      source.Tofile(path);

      Storage loaded = Storage::Fromfile(path, Type.Double, 0);
      EXPECT_EQ(loaded.dtype(), Type.Double);
      EXPECT_EQ(loaded.size(), 0);
      EXPECT_EQ(loaded.clone().size(), 0);
      EXPECT_EQ(loaded.astype(Type.Float).dtype(), Type.Float);
      EXPECT_TRUE(loaded.vector<cytnx_double>().empty());
    }

    TEST_F(StorageTest, InitByPtrRejectsZeroLength) {
      Storage storage(1, Type.Double);
      cytnx_double value = 1.0;

      EXPECT_THROW(storage._impl->_Init_byptr(&value, 0, Device.cpu), std::logic_error);
    }

    TEST_F(StorageTest, FromfileCountEqualsTotalElementsReadsAll) {
      const std::string path = ::testing::TempDir() + "cytnx_storage_fromfile_count_exact.bin";
      std::remove(path.c_str());
      const RemoveFileOnExit cleanup(path);

      Storage source = Storage::from_vector(std::vector<cytnx_double>{1.0, 2.0, 3.0, 4.0});
      source.Tofile(path);

      // count == total elements: explicit count should match count < 0 behaviour
      Storage loaded_explicit = Storage::Fromfile(path, Type.Double, 4);
      Storage loaded_default = Storage::Fromfile(path, Type.Double, -1);
      ASSERT_EQ(loaded_explicit.size(), 4);
      ASSERT_EQ(loaded_default.size(), 4);
      for (cytnx_uint64 i = 0; i < 4; i++)
        EXPECT_DOUBLE_EQ(loaded_explicit.at<cytnx_double>(i), loaded_default.at<cytnx_double>(i));
    }

    TEST_F(StorageTest, AtDtypeMismatchThrows) {
      Storage s(4, Type.Float);
      EXPECT_THROW(s.at<double>(0), std::logic_error);
      EXPECT_THROW(s.at<cytnx_int64>(0), std::logic_error);
      EXPECT_NO_THROW(s.at<float>(0));
    }

    TEST_F(StorageTest, AtOutOfBoundMessageReportsIndexAndSize) {
      Storage s(4, Type.Double);
      try {
        s.at<cytnx_double>(5);
        FAIL() << "expected out-of-bound access to throw";
      } catch (const std::logic_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("[5]"), std::string::npos) << msg;
        EXPECT_NE(msg.find("[4]"), std::string::npos) << msg;
      }
    }

    TEST_F(StorageTest, AtDtypeMismatchThrowsForAllDtypes) {
      for (auto dt : test::dtype_list) {
        Storage s(2, dt);
        const std::string name = Type.getname(dt);
        // Every mismatching at<T> specialization must throw, so that no single
        // specialization can silently lose its dtype check again.
        if (dt != Type.ComplexDouble)
          EXPECT_THROW(s.at<cytnx_complex128>(0), std::logic_error) << "storage dtype " << name;
        if (dt != Type.ComplexFloat)
          EXPECT_THROW(s.at<cytnx_complex64>(0), std::logic_error) << "storage dtype " << name;
        if (dt != Type.Double)
          EXPECT_THROW(s.at<cytnx_double>(0), std::logic_error) << "storage dtype " << name;
        if (dt != Type.Float)
          EXPECT_THROW(s.at<cytnx_float>(0), std::logic_error) << "storage dtype " << name;
        if (dt != Type.Int64)
          EXPECT_THROW(s.at<cytnx_int64>(0), std::logic_error) << "storage dtype " << name;
        if (dt != Type.Uint64)
          EXPECT_THROW(s.at<cytnx_uint64>(0), std::logic_error) << "storage dtype " << name;
        if (dt != Type.Int32)
          EXPECT_THROW(s.at<cytnx_int32>(0), std::logic_error) << "storage dtype " << name;
        if (dt != Type.Uint32)
          EXPECT_THROW(s.at<cytnx_uint32>(0), std::logic_error) << "storage dtype " << name;
        if (dt != Type.Int16)
          EXPECT_THROW(s.at<cytnx_int16>(0), std::logic_error) << "storage dtype " << name;
        if (dt != Type.Uint16)
          EXPECT_THROW(s.at<cytnx_uint16>(0), std::logic_error) << "storage dtype " << name;
        if (dt != Type.Bool)
          EXPECT_THROW(s.at<cytnx_bool>(0), std::logic_error) << "storage dtype " << name;
      }
    }

    // --- #941 typed storage dispatch foundation: Type.hpp helpers ---

    TEST(TypePromotion, MakeFloatingPointIntegralToDouble) {
      static_assert(std::is_same_v<Type_class::make_floating_point_t<cytnx_int16>, cytnx_double>);
      static_assert(std::is_same_v<Type_class::make_floating_point_t<cytnx_uint64>, cytnx_double>);
      static_assert(std::is_same_v<Type_class::make_floating_point_t<cytnx_bool>, cytnx_double>);
      SUCCEED();
    }

    TEST(TypePromotion, MakeFloatingPointFloatingUnchanged) {
      static_assert(std::is_same_v<Type_class::make_floating_point_t<cytnx_double>, cytnx_double>);
      static_assert(std::is_same_v<Type_class::make_floating_point_t<cytnx_float>, cytnx_float>);
      SUCCEED();
    }

    TEST(TypePromotion, MakeFloatingPointComplexUnchanged) {
      static_assert(
        std::is_same_v<Type_class::make_floating_point_t<cytnx_complex128>, cytnx_complex128>);
      static_assert(
        std::is_same_v<Type_class::make_floating_point_t<cytnx_complex64>, cytnx_complex64>);
      SUCCEED();
    }

    TEST(TypePromotion, MakeFloatingPointRuntimeMatchesCompileTime) {
      EXPECT_EQ(Type_class::make_floating_point_dtype(Type.Int16), (unsigned int)Type.Double);
      EXPECT_EQ(Type_class::make_floating_point_dtype(Type.Uint64), (unsigned int)Type.Double);
      EXPECT_EQ(Type_class::make_floating_point_dtype(Type.Bool), (unsigned int)Type.Double);
      EXPECT_EQ(Type_class::make_floating_point_dtype(Type.Double), (unsigned int)Type.Double);
      EXPECT_EQ(Type_class::make_floating_point_dtype(Type.Float), (unsigned int)Type.Float);
      EXPECT_EQ(Type_class::make_floating_point_dtype(Type.ComplexDouble),
                (unsigned int)Type.ComplexDouble);
      EXPECT_EQ(Type_class::make_floating_point_dtype(Type.ComplexFloat),
                (unsigned int)Type.ComplexFloat);
    }

    // --- #941 typed storage dispatch foundation: StorageImplementation<T> typed accessors ---

    TEST(StorageImplementationTyped, ValueTypeAndDtypeValue) {
      static_assert(std::is_same_v<StorageImplementation<cytnx_double>::value_type, cytnx_double>);
      static_assert(StorageImplementation<cytnx_double>::dtype_value == Type.Double);
      static_assert(StorageImplementation<cytnx_int16>::dtype_value == Type.Int16);
      SUCCEED();
    }

    TEST(StorageImplementationTyped, TypedDataPointer) {
      Storage s(5, Type.Double);
      auto* impl = dynamic_cast<StorageImplementation<cytnx_double>*>(s._impl.get());
      ASSERT_NE(impl, nullptr);
      cytnx_double* p = impl->data();  // must be cytnx_double*, not void*
      for (int i = 0; i < 5; i++) p[i] = static_cast<cytnx_double>(i);
      EXPECT_DOUBLE_EQ(s.at<cytnx_double>(0), 0.0);
      EXPECT_DOUBLE_EQ(s.at<cytnx_double>(4), 4.0);
    }

    // --- #941 typed storage dispatch foundation: StorageVariant / as_storage_variant ---

    TEST(StorageVariantTest, AsStorageVariantRoundTrip) {
      Storage s = Storage::from_vector<cytnx_int32>({1, 2, 3});
      StorageVariant v = s.as_storage_variant();
      bool visited = false;
      std::visit(
        [&](auto impl) {
          using T = storage_value_t<decltype(impl)>;
          // std::visit instantiates this lambda body for every alternative in
          // StorageVariant, not just the active one -- guard the dtype-specific
          // assertion with if constexpr so only the actually-active Int32
          // alternative runs it at runtime (the others compile but no-op).
          if constexpr (std::is_same_v<T, cytnx_int32>) {
            EXPECT_EQ(impl->size(), 3u);
            EXPECT_EQ(impl->data()[0], 1);
            visited = true;
          }
        },
        v);
      EXPECT_TRUE(visited);
    }

    TEST(StorageVariantTest, AsStorageVariantVoidThrows) {
      Storage s;  // default-constructed Storage_base (dtype Void)
      EXPECT_THROW(s.as_storage_variant(), std::logic_error);
    }

    TEST(StorageVariantTest, VariantSizeMatchesNonVoidTypeList) {
      // 11 non-void alternatives, mirrors Type_list minus void -- same pattern as
      // T2's Scalar static_asserts tying its variant to Type_list.
      static_assert(std::variant_size_v<StorageVariant> == std::variant_size_v<Type_list> - 1);
    }

    // --- #941 typed storage dispatch foundation: storage_as_type_or_replace ---

    TEST(StorageAsTypeOrReplace, ReusesMatchingBuffer) {
      Storage out = Storage::from_vector<cytnx_double>({1.0, 2.0, 3.0});
      auto* original_impl = out._impl.get();
      auto typed = storage_as_type_or_replace<cytnx_double>(out, 3, Device.cpu);
      EXPECT_EQ(static_cast<Storage_base*>(typed.get()), original_impl);  // same object, reused
      EXPECT_EQ(typed->size(), 3u);
    }

    TEST(StorageAsTypeOrReplace, ReplacesOnTypeMismatch) {
      Storage out = Storage::from_vector<cytnx_int32>({1, 2, 3});
      auto typed = storage_as_type_or_replace<cytnx_double>(out, 3, Device.cpu);
      EXPECT_EQ(out.dtype(), (unsigned int)Type.Double);
      EXPECT_EQ(typed->size(), 3u);
    }

    TEST(StorageAsTypeOrReplace, ReplacesOnSizeMismatch) {
      Storage out = Storage::from_vector<cytnx_double>({1.0, 2.0, 3.0});
      auto typed = storage_as_type_or_replace<cytnx_double>(out, 5, Device.cpu);
      EXPECT_EQ(typed->size(), 5u);
      EXPECT_EQ(out.size(), 5u);
    }

    TEST(StorageAsTypeOrReplace, AllocatesWhenOutIsEmpty) {
      Storage out;  // Type.Void
      auto typed = storage_as_type_or_replace<cytnx_float>(out, 4, Device.cpu);
      EXPECT_EQ(out.dtype(), (unsigned int)Type.Float);
      EXPECT_EQ(typed->size(), 4u);
    }

    // --- #941 ruling 2: capacity_ removed; append() = exact realloc (O(n), numpy-like) ---

    TEST(StorageCapacityRemoval, AppendGrowsExactly) {
      Storage s = Storage::from_vector<cytnx_int32>({1, 2, 3});
      EXPECT_EQ(s.size(), 3u);
      s.append(cytnx_int32(4));
      EXPECT_EQ(s.size(), 4u);
      EXPECT_EQ(s.at<cytnx_int32>(3), 4);
      s.append(cytnx_int32(5));
      EXPECT_EQ(s.size(), 5u);
      EXPECT_EQ(s.at<cytnx_int32>(4), 5);
    }

    TEST(StorageCapacityRemoval, AppendPreservesExistingElements) {
      Storage s = Storage::from_vector<cytnx_double>({1.0, 2.0, 3.0});
      s.append(cytnx_double(4.0));
      for (int i = 0; i < 3; i++) EXPECT_DOUBLE_EQ(s.at<cytnx_double>(i), (double)(i + 1));
      EXPECT_DOUBLE_EQ(s.at<cytnx_double>(3), 4.0);
    }

    TEST(StorageCapacityRemoval, ResizeShrinkAndGrowPreservesPrefix) {
      Storage s = Storage::from_vector<cytnx_int64>({10, 20, 30, 40});
      s.resize(2);
      EXPECT_EQ(s.size(), 2u);
      EXPECT_EQ(s.at<cytnx_int64>(0), 10);
      EXPECT_EQ(s.at<cytnx_int64>(1), 20);
      s.resize(5);
      EXPECT_EQ(s.size(), 5u);
      EXPECT_EQ(s.at<cytnx_int64>(0), 10);
      EXPECT_EQ(s.at<cytnx_int64>(1), 20);
    }

  }  // namespace test
}  // namespace cytnx
