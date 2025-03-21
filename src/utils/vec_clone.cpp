#include "utils/vec_clone.hpp"

#include "Symmetry.hpp"
#include "Bond.hpp"

#ifdef BACKEND_TORCH
#else

  #include "Tensor.hpp"
  #include "backend/Storage.hpp"
  #include "UniTensor.hpp"

#endif  // BACKEND_TORCH

namespace cytnx {

  template <class T>
  std::vector<T> vec_clone(const std::vector<T> &in_vec) {
    std::vector<T> out(in_vec.size());
    for (cytnx_uint64 i = 0; i < in_vec.size(); i++) {
      out[i] = in_vec[i].clone();
    }
    return out;
  }
  //=======================================================

  template <>
  std::vector<std::string> vec_clone(const std::vector<std::string> &in_vec,
                                     const std::vector<cytnx_uint64> &locators) {
    std::vector<std::string> out(locators.size());
    for (cytnx_uint64 i = 0; i < locators.size(); i++) {
      cytnx_error_msg(locators[i] >= in_vec.size(),
                      "[ERROR] the index [%d] in locators exceed the bbound.\n", locators[i]);
      out[i] = in_vec[locators[i]];
    }
    return out;
  }

  template <>
  std::vector<cytnx_uint64> vec_clone(const std::vector<cytnx_uint64> &in_vec,
                                      const std::vector<cytnx_uint64> &locators) {
    std::vector<cytnx_uint64> out(locators.size());
    for (cytnx_uint64 i = 0; i < locators.size(); i++) {
      cytnx_error_msg(locators[i] >= in_vec.size(),
                      "[ERROR] the index [%d] in locators exceed the bbound.\n", locators[i]);
      out[i] = in_vec[locators[i]];
    }
    return out;
  }

#ifdef BACKEND_TORCH
#else

  template std::vector<Tensor> vec_clone<Tensor>(const std::vector<Tensor> &);
  template std::vector<Storage> vec_clone<Storage>(const std::vector<Storage> &);
  template std::vector<UniTensor> vec_clone<UniTensor>(const std::vector<UniTensor> &);

#endif

  template std::vector<Bond> vec_clone<Bond>(const std::vector<Bond> &);
  template std::vector<Symmetry> vec_clone<Symmetry>(const std::vector<Symmetry> &);

}  // namespace cytnx
