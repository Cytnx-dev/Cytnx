#include <algorithm>
#include <complex>
#include <limits>
#include <vector>

#include "gtest/gtest.h"

#include "linalg.hpp"
#include "Tensor.hpp"
#include "Type.hpp"

namespace cytnx {
  namespace test {
    namespace {

      template <typename T>
      Tensor MakeVector(const std::vector<T>& values) {
        Tensor out({values.size()}, Type.cy_typeid_v<T>);
        std::copy(values.begin(), values.end(), out.ptr_as<T>());
        return out;
      }

      template <typename T>
      void ExpectScalar(const Tensor& result, T expected) {
        EXPECT_EQ(result.dtype(), Type.cy_typeid_v<T>);
        EXPECT_EQ(result.rank(), 0u);
        EXPECT_EQ(result.item<T>(), expected);
      }

      template <typename T>
      class LinalgMaxMinSignedTest : public ::testing::Test {};
      using SignedRealTypes =
        ::testing::Types<cytnx_double, cytnx_float, cytnx_int64, cytnx_int32, cytnx_int16>;
      TYPED_TEST_SUITE(LinalgMaxMinSignedTest, SignedRealTypes);

      // The running maximum used to start from the smallest *positive* value (DBL_MIN / FLT_MIN),
      // so an all-negative floating-point tensor returned that start value.
      TYPED_TEST(LinalgMaxMinSignedTest, AllNegative) {
        using T = TypeParam;
        const Tensor tensor = MakeVector<T>({T(-3), T(-1), T(-7), T(-2)});
        ExpectScalar<T>(linalg::Max(tensor), T(-1));
        ExpectScalar<T>(linalg::Min(tensor), T(-7));
      }

      TYPED_TEST(LinalgMaxMinSignedTest, MixedSign) {
        using T = TypeParam;
        const Tensor tensor = MakeVector<T>({T(4), T(-9), T(0), T(6), T(-2)});
        ExpectScalar<T>(linalg::Max(tensor), T(6));
        ExpectScalar<T>(linalg::Min(tensor), T(-9));
      }

      TYPED_TEST(LinalgMaxMinSignedTest, SingleElement) {
        using T = TypeParam;
        const Tensor tensor = MakeVector<T>({T(-5)});
        ExpectScalar<T>(linalg::Max(tensor), T(-5));
        ExpectScalar<T>(linalg::Min(tensor), T(-5));
      }

      // The reduction runs over every element whatever the shape, and the result is rank 0.
      TYPED_TEST(LinalgMaxMinSignedTest, Rank3) {
        using T = TypeParam;
        const Tensor tensor = MakeVector<T>({T(-8), T(-3), T(-6), T(-11), T(-2), T(-9), T(-4),
                                             T(-12), T(-5), T(-7), T(-10), T(-13)})
                                .reshape({2, 3, 2});
        ExpectScalar<T>(linalg::Max(tensor), T(-2));
        ExpectScalar<T>(linalg::Min(tensor), T(-13));
      }

      template <typename T>
      class LinalgMaxMinUnsignedTest : public ::testing::Test {};
      using UnsignedTypes = ::testing::Types<cytnx_uint64, cytnx_uint32, cytnx_uint16>;
      TYPED_TEST_SUITE(LinalgMaxMinUnsignedTest, UnsignedTypes);

      TYPED_TEST(LinalgMaxMinUnsignedTest, Values) {
        using T = TypeParam;
        const Tensor tensor = MakeVector<T>({T(4), T(9), T(2), T(6)});
        ExpectScalar<T>(linalg::Max(tensor), T(9));
        ExpectScalar<T>(linalg::Min(tensor), T(2));
      }

      template <typename T>
      class LinalgMaxMinFloatingTest : public ::testing::Test {};
      using FloatingTypes = ::testing::Types<cytnx_double, cytnx_float>;
      TYPED_TEST_SUITE(LinalgMaxMinFloatingTest, FloatingTypes);

      TYPED_TEST(LinalgMaxMinFloatingTest, FractionalNegative) {
        using T = TypeParam;
        const Tensor tensor = MakeVector<T>({T(-3.5), T(-1.25), T(-7.75)});
        ExpectScalar<T>(linalg::Max(tensor), T(-1.25));
        ExpectScalar<T>(linalg::Min(tensor), T(-7.75));
      }

      TYPED_TEST(LinalgMaxMinFloatingTest, Infinite) {
        using T = TypeParam;
        const T inf = std::numeric_limits<T>::infinity();
        ExpectScalar<T>(linalg::Max(MakeVector<T>({-inf, -inf})), -inf);
        ExpectScalar<T>(linalg::Min(MakeVector<T>({inf, inf})), inf);
        ExpectScalar<T>(linalg::Max(MakeVector<T>({T(1), inf, T(-2)})), inf);
        ExpectScalar<T>(linalg::Min(MakeVector<T>({T(1), -inf, T(-2)})), -inf);
      }

      template <typename T>
      class LinalgMaxMinComplexTest : public ::testing::Test {};
      using ComplexTypes = ::testing::Types<cytnx_complex128, cytnx_complex64>;
      TYPED_TEST_SUITE(LinalgMaxMinComplexTest, ComplexTypes);

      // Complex elements are ordered by their real part; the imaginary part is carried along.
      TYPED_TEST(LinalgMaxMinComplexTest, AllNegativeRealPart) {
        using T = TypeParam;
        const Tensor tensor = MakeVector<T>({T(-3.5, 1), T(-1.25, -2), T(-7.75, 4)});
        ExpectScalar<T>(linalg::Max(tensor), T(-1.25, -2));
        ExpectScalar<T>(linalg::Min(tensor), T(-7.75, 4));
      }

      TYPED_TEST(LinalgMaxMinComplexTest, OrdersByRealPartOnly) {
        using T = TypeParam;
        const Tensor tensor = MakeVector<T>({T(1, 50), T(3, -8), T(-2, 90), T(2, 0)});
        ExpectScalar<T>(linalg::Max(tensor), T(3, -8));
        ExpectScalar<T>(linalg::Min(tensor), T(-2, 90));
      }

      // Among elements with equal real part the last one is returned.
      TYPED_TEST(LinalgMaxMinComplexTest, EqualRealPartReturnsLast) {
        using T = TypeParam;
        ExpectScalar<T>(linalg::Max(MakeVector<T>({T(1, 5), T(2, 1), T(2, -3), T(0, 9)})),
                        T(2, -3));
        ExpectScalar<T>(linalg::Min(MakeVector<T>({T(1, 5), T(0, 1), T(0, -3), T(4, 9)})),
                        T(0, -3));
      }

      TYPED_TEST(LinalgMaxMinComplexTest, InfiniteRealPart) {
        using T = TypeParam;
        const auto inf = std::numeric_limits<typename T::value_type>::infinity();
        ExpectScalar<T>(linalg::Max(MakeVector<T>({T(-inf, 2), T(-inf, 3)})), T(-inf, 3));
        ExpectScalar<T>(linalg::Min(MakeVector<T>({T(inf, 2), T(inf, 3)})), T(inf, 3));
      }

    }  // namespace
  }  // namespace test
}  // namespace cytnx
