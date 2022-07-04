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

  /* 2.
      create a non-symmetry bond tagged with BD_KET
      with dimension 10
  */
  Bond bd_b = Bond(10, BD_KET);
  cout << bd_b << endl;

  /* 3.
      crate a symmetry bond,
      with single U1 (default) symmetry and qnums = (0,2,-1,3)
  */
  Bond bd_c = Bond(4, BD_KET, {{0}, {2}, {-1}, {3}});
  cout << bd_c << endl;

  /* 3.
      crate a symmetry bond,
      with U1 x Z2 multiple symmetry
      and qnums = U1:(0,2,-1,3) x Z2:(0,1,1,0)
  */
  Bond bd_d = Bond(4, BD_BRA, {{0, 0}, {2, 1}, {-1, 1}, {3, 0}}, {Symmetry::U1(), Symmetry::Zn(2)});
  cout << bd_d << endl;
}
