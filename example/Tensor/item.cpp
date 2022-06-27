#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
using namespace std;
int main() {
  Tensor A = ones(1, Type.Uint64);
  cout << A << endl;

  // note that type resolver should be consist with the dtype
  cout << A.item<cytnx_uint64>() << endl;

  return 0;
}
