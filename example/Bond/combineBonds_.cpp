#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
int main() {
  /* 1.
      combine multiple KET bonds
  */
  Bond bd_a = Bond(10, BD_KET);
  Bond bd_b = Bond(4, BD_KET);
  Bond bd_c = Bond(5, BD_KET);
  Bond bd_d = Bond(2, BD_KET);
  std::cout << bd_a << std::endl;
  std::cout << bd_b << std::endl;
  std::cout << bd_c << std::endl;
  std::cout << bd_d << std::endl;

  bd_a.combineBonds_({bd_b, bd_c, bd_d});
  std::cout << bd_a << std::endl;

  /* 2.
      combine symmetry bonds,
      with U1 x Z2 multiple symmetry
  */
  Bond bd_sym_a =
    Bond(BD_BRA, {Qs(0, 1) >> 1, Qs(2, 0) >> 1, Qs(-4, 1) >> 1}, {Symmetry::U1(), Symmetry::Zn(2)});

  Bond bd_sym_b = Bond(BD_BRA, {Qs(0, 0) >> 1, Qs(2, 1) >> 1, Qs(-1, 1) >> 1, Qs(3, 0) >> 1},
                       {Symmetry::U1(), Symmetry::Zn(2)});

  Bond bd_sym_c = Bond(BD_BRA, {Qs(1, 1) >> 2, Qs(-1, 1) >> 1, Qs(-2, 0) >> 1, Qs(0, 0) >> 1},
                       {Symmetry::U1(), Symmetry::Zn(2)});

  std::cout << bd_sym_a << std::endl;
  std::cout << bd_sym_b << std::endl;
  std::cout << bd_sym_c << std::endl;
  bd_sym_a.combineBonds_({bd_sym_b, bd_sym_c});
  std::cout << bd_sym_a << std::endl;

  return 0;
}
