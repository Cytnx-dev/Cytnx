#ifndef CYTNX_TENSORT_TRAITS_HPP_
#define CYTNX_TENSORT_TRAITS_HPP_

#include "TensorT.hpp"
#include "Type.hpp"

#include <concepts>
#include <tuple>
#include <type_traits>
#include <variant>

namespace cytnx {

  /// Element type concept for the supported real floating-point TensorT scalar types.
  template <class T>
  concept RealScalar = std::same_as<std::remove_cv_t<T>, cytnx_float> ||
    std::same_as<std::remove_cv_t<T>, cytnx_double>;

  /// Element type concept for the supported complex floating-point TensorT scalar types.
  template <class T>
  concept ComplexScalar = std::same_as<std::remove_cv_t<T>, cytnx_complex64> ||
    std::same_as<std::remove_cv_t<T>, cytnx_complex128>;

  /// Element type concept for supported real or complex floating-point TensorT scalar types.
  template <class T>
  concept NumericScalar = RealScalar<T> || ComplexScalar<T>;

  /// Type list used to build real TensorT dispatch variants.
  using RealScalars = std::tuple<cytnx_float, cytnx_double>;

  /// Type list used to build complex TensorT dispatch variants.
  using ComplexScalars = std::tuple<cytnx_complex64, cytnx_complex128>;

  /// Type list used to build real-or-complex TensorT dispatch variants.
  using NumericScalars = std::tuple<cytnx_float, cytnx_double, cytnx_complex64, cytnx_complex128>;

#ifdef UNI_GPU
  using TensorAccesses = std::tuple<host_access, cuda_access>;
#else
  using TensorAccesses = std::tuple<host_access>;
#endif

  namespace tensor_t_detail {

    template <std::size_t Rank, class Layout, class Scalar, class Accesses>
    struct tensor_variant_for_scalar;

    template <std::size_t Rank, class Layout, class Scalar, class... Accesses>
    struct tensor_variant_for_scalar<Rank, Layout, Scalar, std::tuple<Accesses...>> {
      using type = std::variant<TensorT<Scalar, Rank, Accesses, Layout>...>;
    };

    template <class... Variants>
    struct variant_cat;

    template <class... Types>
    struct variant_cat<std::variant<Types...>> {
      using type = std::variant<Types...>;
    };

    template <class... Left, class... Right, class... Rest>
    struct variant_cat<std::variant<Left...>, std::variant<Right...>, Rest...> {
      using type = typename variant_cat<std::variant<Left..., Right...>, Rest...>::type;
    };

    template <std::size_t Rank, class Layout, class Scalars, class Accesses>
    struct tensor_variant_from_lists;

    template <std::size_t Rank, class Layout, class... Scalars, class... Accesses>
    struct tensor_variant_from_lists<Rank, Layout, std::tuple<Scalars...>,
                                     std::tuple<Accesses...>> {
      using type = typename variant_cat<typename tensor_variant_for_scalar<
        Rank, Layout, Scalars, std::tuple<Accesses...>>::type...>::type;
    };

  }  // namespace tensor_t_detail

  template <std::size_t Rank, class Layout, class Scalars, class Accesses = TensorAccesses>
  using TensorVariantT =
    typename tensor_t_detail::tensor_variant_from_lists<Rank, Layout, Scalars, Accesses>::type;

  /// Variant over real scalar types and enabled access backends.
  template <std::size_t Rank, class Layout = stdex::layout_right>
  using RealTensor = TensorVariantT<Rank, Layout, RealScalars>;

  /// Variant over complex scalar types and enabled access backends.
  template <std::size_t Rank, class Layout = stdex::layout_right>
  using ComplexTensor = TensorVariantT<Rank, Layout, ComplexScalars>;

  /// Variant over real-or-complex scalar types and enabled access backends.
  template <std::size_t Rank, class Layout = stdex::layout_right>
  using NumericTensor = TensorVariantT<Rank, Layout, NumericScalars>;

  namespace tensor_t_detail {

    template <class T>
    struct tensor_t_alternative;

    template <class T, std::size_t Rank, class Access, class Layout>
    struct tensor_t_alternative<TensorT<T, Rank, Access, Layout>> {
      using element_type = T;
      using access_type = Access;
      using layout_type = Layout;
      static constexpr std::size_t rank = Rank;
    };

    template <class Alternative>
    bool tensor_matches_alternative(const Tensor &tensor) {
      using traits = tensor_t_alternative<Alternative>;
      using element_type = typename traits::element_type;
      using access_type = typename traits::access_type;
      return tensor.rank() == traits::rank &&
             tensor.dtype() == Type_class::cy_typeid_v<std::remove_cv_t<element_type>> &&
             access_accepts_device(access_type{}, tensor.device());
    }

    template <class Alternative, class Variant>
    bool try_make_tensor_alternative(const Tensor &input, Variant &out) {
      if (!tensor_matches_alternative<Alternative>(input)) return false;

      using traits = tensor_t_alternative<Alternative>;
      using element_type = typename traits::element_type;
      using access_type = typename traits::access_type;
      using layout_type = typename traits::layout_type;

      if constexpr (std::same_as<layout_type, stdex::layout_right>) {
        Tensor tensor = input.contiguous();
        out = make_right_tensor_t<element_type, traits::rank, access_type>(tensor);
      } else if constexpr (std::same_as<layout_type, stdex::layout_stride>) {
        Tensor tensor = input;
        out = make_tensor_t<element_type, traits::rank, access_type>(tensor);
      } else {
        static_assert(std::same_as<layout_type, stdex::layout_right> ||
                        std::same_as<layout_type, stdex::layout_stride>,
                      "Unsupported TensorT layout for make_tensor");
      }
      return true;
    }

    template <class Variant, std::size_t... Indices>
    Variant make_tensor_impl(const Tensor &tensor, std::index_sequence<Indices...>) {
      Variant out;
      const bool matched =
        (try_make_tensor_alternative<std::variant_alternative_t<Indices, Variant>>(tensor, out) ||
         ...);
      cytnx_error_msg(!matched,
                      "[ERROR] Tensor dtype/device/rank is not represented by this TensorT "
                      "variant.%s",
                      "\n");
      return out;
    }

  }  // namespace tensor_t_detail

  /**
   * @brief Create a typed TensorT variant from a legacy Tensor.
   *
   * `Variant` must be a `std::variant` whose alternatives are `TensorT` specializations. The
   * factory selects the alternative matching the Tensor rank, dtype, device backend, and requested
   * layout. Layout-right alternatives are made contiguous without mutating the input Tensor.
   */
  template <class Variant>
  Variant make_tensor(const Tensor &tensor) {
    return tensor_t_detail::make_tensor_impl<Variant>(
      tensor, std::make_index_sequence<std::variant_size_v<Variant>>{});
  }

}  // namespace cytnx

#endif  // CYTNX_TENSORT_TRAITS_HPP_
