#ifndef CYTNX_UTILS_PRINT_H_
#define CYTNX_UTILS_PRINT_H_

#include "Type.hpp"
#include "Tensor.hpp"

#include "Bond.hpp"
#include "Symmetry.hpp"
#include "UniTensor.hpp"
#include "Network.hpp"
#include <iostream>

#include "backend/Storage.hpp"
#include "backend/Scalar.hpp"

namespace cytnx {

  template <class T>
  void print(const T &ipt) {
    std::cout << ipt << std::endl;
  }

}  // namespace cytnx

#endif  // CYTNX_UTILS_PRINT_H_
