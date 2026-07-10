#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
int main() {
  Tensor A({3, 4, 5});

  Tensor B = A;  // B shares same object with A
  Tensor C = A.clone();  // C is a copy of A

  // use is() to check if two variable shares same object
  std::cout << is(B, A) << std::endl;
  std::cout << is(C, A) << std::endl;

  return 0;
}
