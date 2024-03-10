#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
using namespace std;
int main() {
  // 1. handy alias:
  typedef Accessor ac;

  /*
      2. Create a Tensor with
      shape (3,4,5),
      dtype =Type.Double [default],
      device=Device.cpu [default]
  */
  Tensor A = cytnx::arange(24);
  A.reshape_({2, 3, 4});
  cout << A << endl;

  /*
      3. Accessing elements using accessor
      This is similar as python slices.
      -> A[0,:,0:2:1]

  */
  Tensor B = A(0, ac::all(), ac::range(0, 2, 1));
  cout << B << endl;

  /* [Note] Conversion from python slice to ac:
          [::x]   = ac::step(x)
          [a::x]  = ac::tilend(a,x)
          [a::]   = ac::tilend(a)
          [:b:]   = ac::range(0,b,1)
          [a:b:x] = ac::range(a,b,x)
  */

  return 0;
}
