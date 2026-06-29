#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
int main() {
  Tensor A = arange(60).reshape({3, 4, 5});
  std::cout << A << std::endl;

  A.fill(999);
  std::cout << A << std::endl;

  return 0;
}
