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

  // Construct an array of typeid(T).name() for each type in Type_list.
  // This is complicated by Type_list containing 'void', which means we can't use an ordinary
  // lambda, but instead we need a metafunction and a template template parameter.
  template <typename T>
  struct c_typename {
    static const char* get() { return typeid(T).name(); }
  };

  template <typename Variant, template <typename> class Func, std::size_t... Indices>
  auto make_type_array_from_func_helper(std::index_sequence<Indices...>) {
    return std::array<decltype(Func<int>::get()), sizeof...(Indices)>{
      Func<std::variant_alternative_t<Indices, Variant>>::get()...};
  }

  template <typename Variant, template <typename> class Func>
  auto make_type_array_from_func() {
    return make_type_array_from_func_helper<Variant, Func>(
      std::make_index_sequence<std::variant_size_v<Variant>>());
  }

  unsigned int Type_class::c_typename_to_id(const std::string& c_name) {
    static auto c_typenames = make_type_array_from_func<Type_list, c_typename>();

    auto i = std::find(c_typenames.begin(), c_typenames.end(), c_name);
    if (i == c_typenames.end()) {
      cytnx_error_msg(true, "[ERROR] typename is not a cytnx type: %s", c_name.c_str());
      return 0;
    }
    return i - c_typenames.begin();
  }

}  // namespace cytnx
