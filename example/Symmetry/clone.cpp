#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
int main() {
  /* 1.
      create symmetry objects
  */
  Symmetry sym_A = Symmetry::U1();
  Symmetry sym_C = Symmetry::U1();

  Symmetry sym_D = sym_A;  // same instance.

  // using is() to check if they are the same instance.
  std::cout << is(sym_D, sym_A) << std::endl;  // true. same instance
  std::cout << (sym_D == sym_A) << std::endl;  // true, same content

  std::cout << is(sym_C, sym_A) << std::endl;  // false. different instance
  std::cout << (sym_C == sym_A) << std::endl;  // true, sane content

  return 0;
}
