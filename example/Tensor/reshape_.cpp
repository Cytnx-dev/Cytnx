#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
int main() {
  Tensor A = arange(60);
  std::cout << A << std::endl;

  // there are two ways you can assign the argument:

  // Method 1: more like 'c++' way:
  A.reshape_({5, 12});  // note the braket{}
  std::cout << A << std::endl;

  // Method 2: more like 'python' way:
  A.reshape_(5, 4, 3);
  std::cout << A << std::endl;

  return 0;
}
