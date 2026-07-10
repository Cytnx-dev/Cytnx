#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
int main() {
  /*
      1. Create a Storage with
      dtype =Type.Double [default],
  */
  Storage A(10);
  std::cout << A.dtype_str() << std::endl;

  Storage B = A;
  Storage C = A.clone();

  std::cout << (B == A) << std::endl;  // true (share same instance)
  std::cout << is(B, A) << std::endl;  // true (share same instance)

  std::cout << (C == A) << std::endl;  // true (the same content.)
  std::cout << is(C, A) << std::endl;  // false (not share same instance)

  return 0;
}
