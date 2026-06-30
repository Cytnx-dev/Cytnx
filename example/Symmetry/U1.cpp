#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
int main() {
  /* 1.
      create a U1 symmetry object
  */
  Symmetry sym_u1 = Symmetry::U1();

  /* 2.
      using with Bond to define a bond with symmetry.
  */
  Bond bd_sym_u1_a = Bond(BD_KET, {{0}, {-4}, {-2}, {3}}, {1, 1, 1, 1}, {sym_u1});
  Bond bd_sym_u1_b = Bond(BD_KET, {{0}, {-4}, {-2}, {3}}, {1, 1, 1, 1});  // default is U1 symmetry
  std::cout << bd_sym_u1_a << std::endl;
  std::cout << bd_sym_u1_b << std::endl;
  std::cout << (bd_sym_u1_a == bd_sym_u1_b) << std::endl;  // true

  Bond bd_sym_u1_c = Bond(BD_KET, {{-1}, {1}, {2}, {-2}, {0}}, {1, 1, 1, 1, 1});
  std::cout << bd_sym_u1_c << std::endl;

  /* 3.
      new qnums will be calculated using Symmetry::combine_rule.
  */
  Bond bd_sym_all = bd_sym_u1_a.combineBond(bd_sym_u1_c);
  std::cout << bd_sym_all << std::endl;

  return 0;
}
