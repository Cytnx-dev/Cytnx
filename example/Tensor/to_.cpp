#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
using namespace std;
int main() {
  Tensor A({3, 4, 5});

  // move the instance tensor to different device
  A.to_(Device.cuda + 0);
  cout << A.device_str() << endl;

  return 0;
}
