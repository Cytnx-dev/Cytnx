#ifndef CYTNX_UTILS_PRINT_H_
#define CYTNX_UTILS_PRINT_H_

#include <iostream>

#include "Bond.hpp"
#include "Network.hpp"
#include "Symmetry.hpp"
#include "Tensor.hpp"
#include "Type.hpp"
#include "UniTensor.hpp"

#ifdef BACKEND_TORCH
#else
  #include "backend/Storage.hpp"
  #include "backend/Scalar.hpp"
#endif  // BACKEND_TORCH

namespace cytnx {

  template <class T>
  void print(const T &ipt) {
    std::cout << ipt << std::endl;
  }

  template <class T>
  void print(std::ostream &os, const T &ipt) {
    os << ipt << std::endl;
  }

}  // namespace cytnx

#endif  // CYTNX_UTILS_PRINT_H_
