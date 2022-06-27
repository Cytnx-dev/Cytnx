#include "cytnx.hpp"
#include <iostream>

using namespace std;
using namespace cytnx;

int main() {
  /* 1.
      create a non-symmetry, regular bond (BD_REG)
      with dimension 10
  */
  Bond bd_a = Bond(10);
  cout << bd_a << endl;

  Bond bd_b = bd_a;
  Bond bd_c = bd_a.clone();

  cout << is(bd_b, bd_a) << endl;  // true, the same instance
  cout << is(bd_c, bd_a) << endl;  // false, different instance

  cout << (bd_b == bd_a) << endl;  // true, same content
  cout << (bd_c == bd_a) << endl;  // true, same content

  return 0;
}
