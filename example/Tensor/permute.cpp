#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
int main() {
  Tensor A({3, 4, 5});
  std::cout << A.shape() << std::endl;

  Tensor B = A.permute({0, 2, 1});
  std::cout << B.shape() << std::endl;

  std::cout << is(B, A) << std::endl;  // this should be false, different object.

  std::cout << B.same_data(A)
            << std::endl;  // this should be true, since no new pointer/memory is created.

  return 0;
}
