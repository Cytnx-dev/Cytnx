#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
int main() {
  typedef Accessor ac;

  /*
      In C++ API, generally, there are two ways you can access Tensor.
      1. Using operator():
          This is more of 'python' way.

      2. Using low-level API or operator[]:
          This is more of 'c++' way.
  */

  Tensor A = arange(60).reshape({3, 4, 5});
  std::cout << A << std::endl;

  // Method 1, Using operator():
  //-----------------------------------------
  Tensor B = A(2, ":", "2:5:1");
  std::cout << B << std::endl;

  /* [Note]
      This is equivalent as:
      > Tensor B = A(2,ac::all(),ac::range(2,5,1));

      See also cytnx::Accessor.
  */

  // Method 2, Using operator[] or low-level API get():
  //----------------------------------------
  Tensor B2 = A[{ac(2), ac::all(), ac::range(2, 5, 1)}];  // remember the {}braket
  std::cout << B2 << std::endl;

  /* [Note]
      You can also use the low-level API get() as
      > Tensor B2 = A.get({ac(2),ac::all(),ac::range(2,5,1)});
  */
  return 0;
}
