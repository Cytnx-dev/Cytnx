#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
using namespace std;
int main() {
  /* 1.
      create symmetry objects
  */
  Symmetry sym_A = Symmetry::U1();
  Symmetry sym_C = Symmetry::U1();

  Symmetry sym_D = sym_A;  // same instance.

  // using is() to check if they are the same instance.
  cout << is(sym_D, sym_A) << endl;  // true. same instance
  cout << (sym_D == sym_A) << endl;  // true, same content

  cout << is(sym_C, sym_A) << endl;  // false. different instance
  cout << (sym_C == sym_A) << endl;  // true, sane content

  return 0;
}
