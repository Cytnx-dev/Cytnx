#include <algorithm>
#include <complex>
#include <type_traits>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "backend/Storage.hpp"
#include "Device.hpp"
#include "random.hpp"
#include "Tensor.hpp"
#include "Type.hpp"

namespace cytnx {

  template <typename Container, typename T>
  struct PushFront;

  template <template <typename...> typename Container, typename T, typename... Ts>
  struct PushFront<Container<Ts...>, T> {
    using type = Container<Ts..., T>;
  };

  template <typename Types, template <typename> typename Predicate>
  struct TypeFilter;

  template <template <typename...> typename Container, template <typename> typename Predicate,
            typename T>
  struct TypeFilter<Container<T>, Predicate> {
    using type = std::conditional_t<Predicate<T>::value, Container<T>, Container<>>;
  };

  template <template <typename...> typename Container, template <typename> typename Predicate,
            typename T, typename... Ts>
  struct TypeFilter<Container<T, Ts...>, Predicate> {
    using Next = typename TypeFilter<Container<Ts...>, Predicate>::type;
    using type = std::conditional_t<Predicate<T>::value, typename PushFront<Next, T>::type, Next>;
  };

  template <typename T>
  struct is_floating_point_layout {
    constexpr static bool value = Type_struct_t<T>::is_float;
  };

  template <typename From, template <typename...> typename To>
  struct ContainerConverterHelper;

  template <template <typename...> typename From, template <typename...> typename To,
            typename... Ts>
  struct ContainerConverterHelper<From<Ts...>, To> {
    using type = To<Ts...>;
  };

  template <typename From, template <typename...> typename To>
  using ContainerConverter = typename ContainerConverterHelper<From, To>::type;

  template <typename Dtype>
  struct MemoryLayoutHelper {
    using type = Dtype;
  };

  template <typename ValueType>
  struct MemoryLayoutHelper<std::complex<ValueType>> {
    using type = ValueType;
  };

  template <typename Dtype>
  using MemoryLayout = typename MemoryLayoutHelper<Dtype>::type;

  using SupportedTypes =
    ContainerConverter<TypeFilter<Type_list, is_floating_point_layout>::type, ::testing::Types>;

  template <typename Dtype>
  class RandomUniform : public ::testing::Test {};

  TYPED_TEST_SUITE(RandomUniform, SupportedTypes);

  using ::testing::Contains;
  using ::testing::Ge;
  using ::testing::Gt;
  using ::testing::Le;
  using ::testing::Lt;
  using ::testing::Not;

  TYPED_TEST(RandomUniform, RespectGivenRange) {
    // (2/3)^50 = 1.57e-9, so there is only a very small chance of failing this test even if giving
    // an arbitrary seed.
    const int kCount = 50;
    const int kCountInValueType = is_complex_v<TypeParam> ? kCount * 2 : kCount;
    // TODO: Stop copying after we apply c++20 or provide begin() and end() for Storage and Tensor
    std::vector<MemoryLayout<TypeParam>> copied_numbers(kCountInValueType);
    MemoryLayout<TypeParam>* start;
    unsigned int dtype = Type_struct_t<TypeParam>::cy_typeid;

    Storage storage(kCount, dtype, Device.cpu);
    random::Make_uniform(storage, /*low=*/-2.0, /*high=*/5.0, /*seed=*/3);
    start = reinterpret_cast<MemoryLayout<TypeParam>*>(storage.data<TypeParam>());
    std::copy(start, start + kCountInValueType, copied_numbers.begin());
    EXPECT_THAT(copied_numbers, Contains(Le(-1.0)));
    EXPECT_THAT(copied_numbers, Not(Contains(Lt(-2.0))));
    EXPECT_THAT(copied_numbers, Contains(Ge(4.0)));
    EXPECT_THAT(copied_numbers, Not(Contains(Gt(5.0))));

    Tensor tensor({kCount}, dtype, Device.cpu);
    random::Make_uniform(tensor, /*low=*/-2.0, /*high=*/5.0, /*seed=*/4);
    start = reinterpret_cast<MemoryLayout<TypeParam>*>(tensor.storage().data<TypeParam>());
    std::copy(start, start + kCountInValueType, copied_numbers.begin());
    EXPECT_THAT(copied_numbers, Contains(Le(-1.0)));
    EXPECT_THAT(copied_numbers, Not(Contains(Lt(-2.0))));
    EXPECT_THAT(copied_numbers, Contains(Ge(4.0)));
    EXPECT_THAT(copied_numbers, Not(Contains(Gt(5.0))));
  }

}  // namespace cytnx
