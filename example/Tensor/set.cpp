#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
using namespace std;
int main() {
  typedef Accessor ac;

  // Let's define two Tensor.
  Tensor A = arange(60).reshape({3, 4, 5});
  cout << A << endl;

  Tensor B = zeros({4, 3});
  cout << B << endl;

  // I. To set part of A with Tesnor B, or constant:
  // [Method 1] Using direct assignment
  //-------------------------------------
  A(2, ":", "2:5:1") = B;
  cout << A << endl;

  A(2, ":", "2:5:1") = 999;
  cout << A << endl;

  // note this is the same as
  // A(ac(2),ac::all(),ac::range(2,5,1)) = B;
  // A[{ac(2),ac::all(),ac::range(2,5,1)}] = B; // note that braket{}

  // [Method 2] Using low-level API set():
  //--------------------------------------
  A.set({ac(2), ac::all(), ac::range(2, 5, 1)}, B);
  cout << A << endl;

  A.set({ac(2), ac::all(), ac::range(0, 2, 1)}, 999);
  cout << A << endl;

  return 0;
}
