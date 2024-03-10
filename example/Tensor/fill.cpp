#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
using namespace std;
int main() {
  Tensor A = arange(60).reshape({3, 4, 5});
  cout << A << endl;

  A.fill(999);
  cout << A << endl;

  return 0;
}
