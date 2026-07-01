#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
int main() {
  Storage A(15);

  Storage B = A;  // B shares same object with A
  Storage C = A.clone();  // C is a copy of A

  // use is() to check if two variable shares same object
  std::cout << is(B, A) << std::endl;
  std::cout << is(C, A) << std::endl;

  return 0;
}
