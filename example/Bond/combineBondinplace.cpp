#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
using namespace std;
int main() {
  /* 1.
      create a non-symmetry, regular bond (BD_REG)
      with dimension 10
  */
  Bond bd_a = Bond(10);
  cout << bd_a << endl;

  Bond bd_b = Bond(5);
  cout << bd_b << endl;

  bd_a.combineBond_(bd_b);
  cout << bd_a << endl;

  /* 2.
      combine symmetry bonds,
      with U1 x Z2 multiple symmetry
  */
  Bond bd_c = Bond(3, BD_BRA, {{0, 1}, {2, 0}, {-4, 1}}, {Symmetry::U1(), Symmetry::Zn(2)});

  cout << bd_c << endl;

  Bond bd_d = Bond(4, BD_BRA, {{0, 0}, {2, 1}, {-1, 1}, {3, 0}}, {Symmetry::U1(), Symmetry::Zn(2)});
  cout << bd_d << endl;

  bd_c.combineBond_(bd_d);
  cout << bd_c << endl;

  return 0;
}
