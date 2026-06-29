#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
int main() {
  Tensor A = ones(1, Type.Uint64);
  std::cout << A << std::endl;

  // note that type resolver should be consist with the dtype
  std::cout << A.item<cytnx_uint64>() << std::endl;

  return 0;
}
