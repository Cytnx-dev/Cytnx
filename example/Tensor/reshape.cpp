#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
int main() {
  Tensor A = arange(60);

  // there are two ways you can give argument to reshape:

  // Method 1: more like 'C++' way:
  Tensor B = A.reshape({5, 12});  // note the braket{}
  std::cout << A << std::endl;
  std::cout << B << std::endl;

  // Method 2: more like 'python' way:
  Tensor B2 = A.reshape(5, 12);

  return 0;
}
