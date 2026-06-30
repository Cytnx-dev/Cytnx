#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
int main() {
  Tensor A = zeros({3, 4, 5}, Type.Double);
  std::cout << A;

  Tensor B = A.astype(Type.Uint64);
  std::cout << B;

  // the new type is the same as current dtype, return self.
  Tensor C = A.astype(Type.Double);
  std::cout << is(C, A) << std::endl;  // this should be true.

  return 0;
}
