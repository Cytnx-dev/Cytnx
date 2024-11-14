#ifndef INCLUDE_TYPE_H_
#define INCLUDE_TYPE_H_
#ifndef INCLUDE_TYPE_H_
#define INCLUDE_TYPE_H_

#include <complex>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include "cytnx_error.hpp"

#define MKL_Complex8 std::complex<float>
#define MKL_Complex16 std::complex<double>

#ifdef BACKEND_TORCH
typedef int32_t blas_int;
#else

  #ifdef UNI_MKL
    #include <mkl.h>
typedef MKL_INT blas_int;
  #else
typedef int32_t blas_int;
  #endif

#endif

// @cond
namespace cytnx {

  template <class T>
  using vec3d = std::vector<std::vector<std::vector<T>>>;

  template <class T>
  using vec2d = std::vector<std::vector<T>>;

  typedef double cytnx_double;
  typedef float cytnx_float;
  typedef uint64_t cytnx_uint64;
  typedef uint32_t cytnx_uint32;
  typedef uint16_t cytnx_uint16;
  typedef int64_t cytnx_int64;
  typedef int32_t cytnx_int32;
  typedef int16_t cytnx_int16;
  typedef size_t cytnx_size_t;
  typedef std::complex<float> cytnx_complex64;
  typedef std::complex<double> cytnx_complex128;
  typedef bool cytnx_bool;

  namespace internal {
    template <class>
    struct is_complex_impl : std::false_type {};

    template <class T>
    struct is_complex_impl<std::complex<T>> : std::true_type {};

  }  // namespace internal

  template <typename T>
  using is_complex = internal::is_complex_impl<std::remove_cv_t<T>>;

  // is_complex_v checks if a data type is of type std::complex
  // usage: is_complex_v<T> returns true or false for a data type T
  template <typename T>
  constexpr bool is_complex_v = is_complex<T>::value;

  /// @cond
  struct __type {
    enum __pybind_type {
      Void,
      ComplexDouble,
      ComplexFloat,
      Double,
      Float,
      Int64,
      Uint64,
      Int32,
      Uint32,
      Int16,
      Uint16,
      Bool,
    };
  };

  struct Type_struct {
    const char* name;  // char* is OK here, it is only ever initialized from a string literal
    const char* enum_name;
    bool is_unsigned;
    bool is_complex;
    bool is_float;
    bool is_int;
    unsigned int typeSize;
  };

  template <typename T>
  struct Type_struct_t {
    static constexpr unsigned int cy_typeid = variant_index_v<T, Type_list>;
#ifdef UNI_GPU
    static constexpr unsigned int cy_typeid_gpu = variant_index_v<T, Type_list_gpu>;
#endif
    static constexpr const char* name = Type_names<T>;
    static constexpr const char* enum_name = Type_enum_name<T>;
    static constexpr bool is_complex = is_complex_v<T>;
    static constexpr bool is_unsigned = std::is_unsigned_v<T>;
    static constexpr bool is_float = std::is_floating_point_v<T> || is_complex_floating_point_v<T>;
    static constexpr bool is_int = std::is_integral_v<T> && !std::is_same_v<T, bool>;
    static constexpr std::size_t typeSize = internal::type_size<T>;

    static constexpr Type_struct construct() {
      return {name, enum_name, is_unsigned, is_complex, is_float, is_int, typeSize};
    }
  };

  namespace internal {
    template <typename Variant, std::size_t... Indices>
    constexpr auto make_type_array_helper(std::index_sequence<Indices...>) {
      return std::array<Type_struct, sizeof...(Indices)>{
        Type_struct_t<std::variant_alternative_t<Indices, Variant>>::construct()...};
    }
    template <typename Variant>
    constexpr auto make_type_array() {
      return make_type_array_helper<Variant>(
        std::make_index_sequence<std::variant_size_v<Variant>>());
    }
  }  // namespace internal

  class Type_class {
   private:
   public:
    // Typeinfos is a std::array<Type_struct> for each type in Type_list
    static constexpr auto Typeinfos = internal::make_type_array<Type_list>();

    template <typename T>
    static constexpr unsigned int cy_typeid_v = variant_index_v<T, Type_list>;

#ifdef UNI_GPU
    template <typename T>
    static constexpr unsigned int cy_typeid_gpu_v = variant_index_v<T, Type_list_gpu>;
#endif

    enum Type : unsigned int {
      Void = cy_typeid_v<void>,
      ComplexDouble = cy_typeid_v<cytnx_complex128>,
      ComplexFloat = cy_typeid_v<cytnx_complex64>,
      Double = cy_typeid_v<cytnx_double>,
      Float = cy_typeid_v<cytnx_float>,
      Int64 = cy_typeid_v<cytnx_int64>,
      Uint64 = cy_typeid_v<cytnx_uint64>,
      Int32 = cy_typeid_v<cytnx_int32>,
      Uint32 = cy_typeid_v<cytnx_uint32>,
      Int16 = cy_typeid_v<cytnx_int16>,
      Uint16 = cy_typeid_v<cytnx_uint16>,
      Bool = cy_typeid_v<cytnx_bool>
    };

    static constexpr void check_type(unsigned int type_id) {
      cytnx_error_msg(type_id >= N_Type, "[ERROR] invalid type_id: %s", type_id);
    }

    // This could be constexpr returning constexpr char*, but there is lots of code that
    // assumes that it returns a std::string and calls getname(n).c_str()
    static std::string getname(unsigned int type_id) {
      check_type(type_id);
      return Typeinfos[type_id].name;
    }
    // This cannot be constexpr as we define it in a .cpp file,
    // and typeid(T).name() is not constexpr
    static unsigned int c_typename_to_id(const std::string& c_name);
    static char const* enum_name(unsigned int type_id) {
      check_type(type_id);
      return Typeinfos[type_id].enum_name;
    }
    static constexpr unsigned int typeSize(unsigned int type_id) {
      check_type(type_id);
      return Typeinfos[type_id].typeSize;
    }
    static constexpr bool is_unsigned(unsigned int type_id) {
      check_type(type_id);
      return Typeinfos[type_id].is_unsigned;
    }
    static constexpr bool is_complex(unsigned int type_id) {
      check_type(type_id);
      return Typeinfos[type_id].is_complex;
    }
    static constexpr bool is_float(unsigned int type_id) {
      check_type(type_id);
      return Typeinfos[type_id].is_float;
    }
    static constexpr bool is_int(unsigned int type_id) {
      check_type(type_id);
      return Typeinfos[type_id].is_int;
    }

    template <class T>
    static constexpr unsigned int cy_typeid(const T& rc) {
      return cy_typeid_v<T>;
    }

    // Find a common type for typeL and typeR
    static constexpr unsigned int type_promote(unsigned int typeL, unsigned int typeR) {
      if (typeL < typeR) {
        if (typeL == 0) return 0;

        if (!is_unsigned(typeR) && is_unsigned(typeL)) {
          return typeL - 1;
        } else {
          return typeL;
        }
      } else {
        if (typeR == 0) return 0;
        if (!is_unsigned(typeL) && is_unsigned(typeR)) {
          return typeR - 1;
        } else {
          return typeR;
        }
      }
    }

    // type metafunction for type promotion
    template <typename TL, typename TR>
    using type_promote_t =
      std::variant_alternative_t<Type_class::type_promote(variant_index_v<TL, Type_list>,
                                                          variant_index_v<TR, Type_list>),
                                 Type_list>;

    // Helper to promote two pointer types (note does _not_ return another pointer type)
    template <typename TL, typename TR>
    struct type_promote_from_pointer {
      using type = void;
    };

    template <typename TL, typename TR>
    struct type_promote_from_pointer<TL*, TR*> {
      using type = type_promote_t<std::decay_t<TL>, std::decay_t<TR>>;
    };

    // helper typedef
    template <typename TL, typename TR>
    using type_promote_from_pointer_t = typename type_promote_from_pointer<TL, TR>::type;

#ifdef UNI_GPU
    // .. and we need a version where TL and TR are GPU device pointers
    template <typename TL, typename TR>
    using type_promote_gpu_t =
      std::variant_alternative_t<Type_class::type_promote(variant_index_v<TL, Type_list_gpu>,
                                                          variant_index_v<TR, Type_list_gpu>),
                                 Type_list_gpu>;

    template <typename TL, typename TR>
    struct type_promote_from_gpu_pointer {
      using type = void;
    };

    template <typename TL, typename TR>
    struct type_promote_from_gpu_pointer<TL*, TR*> {
      using type = type_promote_gpu_t<std::decay_t<TL>, std::decay_t<TR>>;
    };

    // helper typedef
    template <typename TL, typename TR>
    using type_promote_from_gpu_pointer_t = typename type_promote_from_gpu_pointer<TL, TR>::type;
#endif

  };  // Type_class
  /// @endcond

  /**
   * @brief data type
   *
   * @details This is the variable about the data type of the UniTensor, Tensor, ... .\n
   *     You can use it as following:
   *     \code
   *     int type = Type.Double;
   *     \endcode
   *
   *     The supported enumerations are as following:
   *
   *  enumeration  |  description
   * --------------|--------------------
   *  Void         |  the data type is void (nothing)
   *  ComplexDouble|  complex double type with 128 bits
   *  ComplexFloat |  complex float type with 64 bits
   *  Double       |  double float type with 64 bits
   *  Float        |  single float type with 32 bits
   *  Int64        |  long long integer type with 64 bits
   *  Uint64       |  unsigned long long integer type with 64 bits
   *  Int32        |  integer type with 32 bits
   *  Uint32       |  unsigned integer type with 32 bits
   *  Int16        |  short integer type with 16 bits
   *  Uint16       |  undigned short integer with 16 bits
   *  Bool         |  boolean type
   */

  constexpr Type_class Type;

  extern int __blasINTsize__;

  extern bool User_debug;

}  // namespace cytnx

#endif  // INCLUDE_TYPE_H_
#endif  // INCLUDE_TYPE_H_
