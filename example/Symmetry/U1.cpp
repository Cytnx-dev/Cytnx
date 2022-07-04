#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
using namespace std;
int main() {
  /* 1.
      create a U1 symmetry object
  */
  Symmetry sym_u1 = Symmetry::U1();

  /* 2.
      using with Bond to define a bond with symmetry.
  */
  Bond bd_sym_u1_a = Bond(4, BD_KET, {{0}, {-4}, {-2}, {3}}, {sym_u1});
  Bond bd_sym_u1_b = Bond(4, BD_KET, {{0}, {-4}, {-2}, {3}});  // default is U1 symmetry
  cout << bd_sym_u1_a << endl;
  cout << bd_sym_u1_b << endl;
  cout << (bd_sym_u1_a == bd_sym_u1_b) << endl;  // true

  Bond bd_sym_u1_c = Bond(5, BD_KET, {{-1}, {1}, {2}, {-2}, {0}});
  cout << bd_sym_u1_c << endl;

  /* 3.
      new qnums will be calculated using Symmetry::combine_rule.
  */
  Bond bd_sym_all = bd_sym_u1_a.combineBond(bd_sym_u1_c);
  cout << bd_sym_all << endl;

  return 0;
}
