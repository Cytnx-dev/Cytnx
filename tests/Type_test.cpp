#include <string>
#include <variant>

#include "gtest/gtest.h"

#include "Generator.hpp"
#include "linalg.hpp"
#include "Tensor.hpp"
#include "Type.hpp"

namespace cytnx {
  namespace test {
    namespace {

#ifdef UNI_GPU
#endif

      static_assert(variant_contains_v<cytnx_double, Type_list>);
      static_assert(variant_contains_v<cytnx_bool, Type_list>);
      static_assert(!variant_contains_v<std::string, Type_list>);

      static_assert(variant_index_v<void, Type_list> == 0);
      static_assert(variant_index_v<cytnx_double, Type_list> == 3);
      static_assert(variant_index_v<cytnx_int64, Type_list> == 5);
      static_assert(variant_index_v<cytnx_bool, Type_list> == 11);

      // to_real/to_complex are constexpr and usable at compile time.
      static_assert(Type_class::to_real(Type_class::ComplexDouble) == Type_class::Double);
      static_assert(Type_class::to_complex(Type_class::Float) == Type_class::ComplexFloat);

      // type_promote's unsigned->signed adjustment (typeX - 1) relies on each
      // unsigned dtype directly following its signed counterpart in Type_list.
      static_assert(Type_class::Uint64 == Type_class::Int64 + 1);
      static_assert(Type_class::Uint32 == Type_class::Int32 + 1);
      static_assert(Type_class::Uint16 == Type_class::Int16 + 1);

#ifdef UNI_GPU
      static_assert(variant_index_v<cytnx_cuda_complex128, Type_list_gpu> ==
                    Type_class::ComplexDouble);
      static_assert(variant_index_v<cytnx_cuda_complex64, Type_list_gpu> ==
                    Type_class::ComplexFloat);
      static_assert(is_complex_v<cytnx_cuda_complex128>);
      static_assert(is_complex_v<cytnx_cuda_complex64>);
      // GPU binary ops (e.g. Kron) promote via the shared, device-independent
      // type_promote_t and map to the CUDA-native type with to_cuda_t at the launch
      // boundary (#1013); the dedicated type_promote_gpu_t trait was retired. Since
      // Type_list_gpu is Type_list with std::complex -> cuda::std::complex (index
      // preserving, asserted above), the promotion is a host value-type property:
      // ComplexFloat x Double -> ComplexDouble.
      static_assert(std::is_same_v<Type_class::type_promote_t<cytnx_complex64, cytnx_double>,
                                   cytnx_complex128>);
      static_assert(std::is_same_v<Type_class::type_promote_t<cytnx_double, cytnx_complex64>,
                                   cytnx_complex128>);
#endif

      TEST(TypeTest, VariantContainsRecognizesSupportedTypes) {
        EXPECT_TRUE((variant_contains_v<cytnx_double, Type_list>));
        EXPECT_TRUE((variant_contains_v<cytnx_bool, Type_list>));
        EXPECT_FALSE((variant_contains_v<std::string, Type_list>));
      }

      TEST(TypeTest, ToRealMapsToRealCounterpart) {
        EXPECT_EQ(Type.to_real(Type.ComplexDouble), Type.Double);
        EXPECT_EQ(Type.to_real(Type.ComplexFloat), Type.Float);
        EXPECT_EQ(Type.to_real(Type.Double), Type.Double);
        EXPECT_EQ(Type.to_real(Type.Int64), Type.Int64);
      }

      TEST(TypeTest, ToComplexMapsToComplexCounterpart) {
        EXPECT_EQ(Type.to_complex(Type.Double), Type.ComplexDouble);
        EXPECT_EQ(Type.to_complex(Type.Float), Type.ComplexFloat);
        EXPECT_EQ(Type.to_complex(Type.ComplexFloat), Type.ComplexFloat);
        EXPECT_EQ(Type.to_complex(Type.Int64), Type.ComplexDouble);
      }

      TEST(TypeTest, PromoteMixedComplexRealUsesMaxPrecision) {
        EXPECT_EQ(Type.type_promote(Type.ComplexFloat, Type.Double), Type.ComplexDouble);
        EXPECT_EQ(Type.type_promote(Type.Double, Type.ComplexFloat), Type.ComplexDouble);
        EXPECT_EQ(Type.type_promote(Type.ComplexFloat, Type.Float), Type.ComplexFloat);
        EXPECT_EQ(Type.type_promote(Type.ComplexDouble, Type.Float), Type.ComplexDouble);
        // unchanged same-kind rules
        EXPECT_EQ(Type.type_promote(Type.Double, Type.Float), Type.Double);
        EXPECT_EQ(Type.type_promote(Type.ComplexFloat, Type.Int64), Type.ComplexFloat);
        EXPECT_EQ(Type.type_promote(Type.Int64, Type.Uint64), Type.Int64);
        // unsigned/signed adjustment: promote to the next-wider signed type
        EXPECT_EQ(Type.type_promote(Type.Uint64, Type.Int32), Type.Int64);
      }

      TEST(TypeTest, TensorAddMixedComplexRealPromotes) {
        auto a = zeros({2}, Type.ComplexFloat);
        auto b = zeros({2}, Type.Double);
        EXPECT_EQ((a + b).dtype(), Type.ComplexDouble);
        EXPECT_EQ((b + a).dtype(), Type.ComplexDouble);
      }

    }  // namespace
  }  // namespace test
}  // namespace cytnx
