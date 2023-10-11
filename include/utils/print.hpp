#ifndef __print__H_
#define __print__H_

#include "Type.hpp"
#include "Tensor.hpp"

#include "Bond.hpp"
#include "Symmetry.hpp"
#include "UniTensor.hpp"
#include "Network.hpp"
#include <iostream>

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

}  // namespace cytnx

#endif
