#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
using namespace std;
int main() {
  /* 1.
      combine multiple KET bonds
  */
  Bond bd_a = Bond(10, BD_KET);
  Bond bd_b = Bond(4, BD_KET);
  Bond bd_c = Bond(5, BD_KET);
  Bond bd_d = Bond(2, BD_KET);

  // note that this will not create copy instances for each Bond elements.
  // so it is both memory efficient and fast!
  Bond bd_all = bd_a.combineBonds({bd_b, bd_c, bd_d});

  cout << bd_a << endl;
  cout << bd_b << endl;
  cout << bd_c << endl;
  cout << bd_d << endl;
  cout << bd_all << endl;

  /* 2.
      combine symmetry bonds,
      with U1 x Z2 multiple symmetry
  */
  Bond bd_sym_a = Bond(3, BD_BRA, {{0, 1}, {2, 0}, {-4, 1}}, {Symmetry::U1(), Symmetry::Zn(2)});

  Bond bd_sym_b =
    Bond(4, BD_BRA, {{0, 0}, {2, 1}, {-1, 1}, {3, 0}}, {Symmetry::U1(), Symmetry::Zn(2)});

  Bond bd_sym_c =
    Bond(5, BD_BRA, {{1, 1}, {1, 1}, {-1, 1}, {-2, 0}, {0, 0}}, {Symmetry::U1(), Symmetry::Zn(2)});

  Bond bd_sym_d = bd_sym_a.combineBonds({bd_sym_b, bd_sym_c});
  cout << bd_sym_a << endl;
  cout << bd_sym_b << endl;
  cout << bd_sym_c << endl;
  cout << bd_sym_d << endl;

  return 0;
}
