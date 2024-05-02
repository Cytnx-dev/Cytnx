#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
using namespace std;
int main() {
  // example of Z2 symmetry object
  //------------------------------------
  //  1. create Z2 object
  Symmetry sym_z2 = Symmetry::Zn(2);

  /* 2.
      using with Bond to define a bond with symmetry.
      Note that the qnums should obey the valid value range of the correspond symmetry.
      with Z2 = [0,2)
  */
  Bond bd_sym_z2_A = Bond(4, BD_KET, {{0}, {0}, {1}, {1}}, {sym_z2});
  Bond bd_sym_z2_B = Bond(3, BD_KET, {{0}, {1}, {1}}, {sym_z2});
  cout << bd_sym_z2_A << endl;
  cout << bd_sym_z2_B << endl;

  /* 3.
      new qnums will be calculated internally using Symmetry::combine_rule.
  */
  Bond bd_sym_z2all = bd_sym_z2_A.combineBond(bd_sym_z2_B);
  cout << bd_sym_z2all << endl;

  // example of Z4 symmetry object
  //------------------------------------
  //  1. create Z4 object
  Symmetry sym_z4 = Symmetry::Zn(4);

  /* 2.
      using with Bond to define a bond with symmetry.
      Note that the qnums should obey the valid value range of the correspond symmetry.
      with Z4 = [0,4)
  */
  Bond bd_sym_z4_A = Bond(4, BD_KET, {{0}, {3}, {1}, {2}}, {sym_z4});
  Bond bd_sym_z4_B = Bond(3, BD_KET, {{2}, {3}, {1}}, {sym_z4});
  cout << bd_sym_z4_A << endl;
  cout << bd_sym_z4_B << endl;

  /* 3.
      new qnums will be calculated internally using Symmetry::combine_rule.
  */
  Bond bd_sym_z4all = bd_sym_z4_A.combineBond(bd_sym_z4_B);
  cout << bd_sym_z4all << endl;

  return 0;
}
