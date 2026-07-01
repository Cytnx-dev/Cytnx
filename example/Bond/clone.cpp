#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;

int main() {
  /* 1.
      create a non-symmetry, regular bond (BD_REG)
      with dimension 10
  */
  Bond bd_a = Bond(10);
  std::cout << bd_a << std::endl;

  Bond bd_b = bd_a;
  Bond bd_c = bd_a.clone();

  std::cout << is(bd_b, bd_a) << std::endl;  // true, the same instance
  std::cout << is(bd_c, bd_a) << std::endl;  // false, different instance

  std::cout << (bd_b == bd_a) << std::endl;  // true, same content
  std::cout << (bd_c == bd_a) << std::endl;  // true, same content

  return 0;
}
