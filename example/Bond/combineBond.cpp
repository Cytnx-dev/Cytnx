#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
using namespace std;
int main() {
  /* 1.
      create a non-symmetry, regular bond (BD_REG)
      with dimension 10
  */
  Bond bd_a = Bond(10, BD_KET);
  Bond bd_b = Bond(15, BD_KET);
  Bond bd_c = bd_a.combineBond(bd_b);
  cout << bd_c << endl;
  cout << bd_a << endl;
  cout << bd_b << endl;

  /* 2.
      combine symmetry bonds,
      with U1 x Z2 multiple symmetry
  */
  Bond bd_d = Bond(BD_BRA, {Qs(0, 1)>>1, Qs(2, 0)>>1, Qs(-4, 1)>>1}, {Symmetry::U1(), Symmetry::Zn(2)});

  Bond bd_e = Bond(BD_BRA, {Qs(0, 0)>>1, Qs(2, 1)>>1, Qs(-1, 1)>>1, Qs(3, 0)>>1}, {Symmetry::U1(), Symmetry::Zn(2)});

  Bond bd_f = bd_d.combineBond(bd_e);
  cout << bd_f << endl;
  cout << bd_d << endl;
  cout << bd_e << endl;
  return 0;
}
