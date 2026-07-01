#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
int main() {
  /* 1.
      create a non-symmetry, regular bond (BD_REG)
      with dimension 10
  */
  Bond bd_a = Bond(10);
  std::cout << bd_a << std::endl;

  /* 2.
      create a non-symmetry bond tagged with BD_IN
      with dimension 10
  */
  Bond bd_b = Bond(10, BD_IN);
  std::cout << bd_b << std::endl;

  /* 3.
      crate a symmetry bond,
      with single U1 (default) symmetry and qnums = (0,2,-1,3)
  */
  Bond bd_c = Bond(BD_IN, {Qs(0) >> 1, Qs(2) >> 1, Qs(-1) >> 1, Qs(3) >> 1});
  std::cout << bd_c << std::endl;

  /* 3.
      crate a symmetry bond,
      with U1 x Z2 multiple symmetry
      and qnums = U1:(0,2,-1,3) x Z2:(0,1,1,0)
  */
  Bond bd_d = Bond(BD_OUT, {Qs(0, 0) >> 1, Qs(2, 1) >> 1, Qs(-1, 1) >> 1, Qs(3, 0) >> 1},
                   {Symmetry::U1(), Symmetry::Zn(2)});
  std::cout << bd_d << std::endl;
}
