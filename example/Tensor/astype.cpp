#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
using namespace std;
int main() {
  Tensor A = zeros({3, 4, 5}, Type.Double);
  cout << A;

  Tensor B = A.astype(Type.Uint64);
  cout << B;

  // the new type is the same as current dtype, return self.
  Tensor C = A.astype(Type.Double);
  cout << is(C, A) << endl;  // this should be true.

  return 0;
}
