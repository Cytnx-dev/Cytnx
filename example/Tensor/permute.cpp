#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
using namespace std;
int main() {
  Tensor A({3, 4, 5});
  cout << A.shape() << endl;

  Tensor B = A.permute({0, 2, 1});
  cout << B.shape() << endl;

  cout << is(B, A) << endl;  // this should be false, different object.

  cout << B.same_data(A) << endl;  // this should be true, since no new pointer/memory is created.

  return 0;
}
