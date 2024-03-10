#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
using namespace std;
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
  cout << A << endl;

  // Method 1, Using operator():
  //-----------------------------------------
  Tensor B = A(2, ":", "2:5:1");
  cout << B << endl;

  /* [Note]
      This is equivalent as:
      > Tensor B = A(2,ac::all(),ac::range(2,5,1));

      See also cytnx::Accessor.
  */

  // Method 2, Using operator[] or low-level API get():
  //----------------------------------------
  Tensor B2 = A[{ac(2), ac::all(), ac::range(2, 5, 1)}];  // remember the {}braket
  cout << B2 << endl;

  /* [Note]
      You can also use the low-level API get() as
      > Tensor B2 = A.get({ac(2),ac::all(),ac::range(2,5,1)});
  */
  return 0;
}
