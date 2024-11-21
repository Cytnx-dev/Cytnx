#include "Type.hpp"
#include "cytnx_error.hpp"
#include <algorithm>

#ifdef BACKEND_TORCH
namespace cytnx {
  int __blasINTsize__ = 32;
}
#else

  #ifdef UNI_MKL
    #include <mkl.h>
namespace cytnx {
  int __blasINTsize__ = sizeof(MKL_INT);
}
  #else
    #include <lapacke.h>
namespace cytnx {
  int __blasINTsize__ = sizeof(lapack_int);
}
  #endif

#endif  // BACKEND_TORCH

using namespace std;

// global debug flag!
namespace cytnx {
  bool User_debug = false;
}

namespace cytnx {
  unsigned int Type_class::type_promote(unsigned int typeL, unsigned int typeR) {
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

  // Construct an array of typeid(T).name() for each type in Type_list.
  // This is complicated by Type_list containing 'void', which means we can't use an ordinary lambda, but
  // instead we need a metafunction and a template template parameter.
  template <typename T>
  struct c_typename {
    static const char* get() {
      return typeid(T).name();
    }
  };

  template <typename Tuple, template <typename> class Func, std::size_t... Indices>
  auto make_type_array_from_func_helper(std::index_sequence<Indices...>) {
    return std::array<decltype(Func<int>::get()), sizeof...(Indices)>{Func<std::tuple_element_t<Indices, Tuple>>::get()...};
  }

  template <typename Tuple, template <typename> class Func>
  auto make_type_array_from_func() {
    return make_type_array_from_func_helper<Tuple, Func>(std::make_index_sequence<std::tuple_size_v<Tuple>>());
  }

  unsigned int Type_class::c_typename_to_id(const std::string &c_name) {
    static auto c_typenames = make_type_array_from_func<Type_list, c_typename>();

    auto i = std::find(c_typenames.begin(), c_typenames.end(), c_name);
    if (i == c_typenames.end()) {
      cytnx_error_msg(true, "[ERROR] typename is not a cytnx type: %s", c_name.c_str());
      return 0;
    }
    return i - c_typenames.begin();
  }

}  // namespace cytnx
