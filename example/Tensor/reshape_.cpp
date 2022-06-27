#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
using namespace std;
int main() {
  Tensor A = arange(60);
  cout << A << endl;

  // there are two ways you can assign the argument:

  // Method 1: more like 'c++' way:
  A.reshape_({5, 12});  // note the braket{}
  cout << A << endl;

  // Method 2: more like 'python' way:
  A.reshape_(5, 4, 3);
  cout << A << endl;

  return 0;
}
