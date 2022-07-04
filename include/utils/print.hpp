#ifndef __print__H_
#define __print__H_

#include "Type.hpp"
#include "Tensor.hpp"
#include "Storage.hpp"
#include "Bond.hpp"
#include "Symmetry.hpp"
#include "UniTensor.hpp"
#include "Network.hpp"
#include "Scalar.hpp"
#include <iostream>

namespace cytnx {

  template <class T>
  void print(const T &ipt) {
    std::cout << ipt << std::endl;
  }

}  // namespace cytnx

#endif
