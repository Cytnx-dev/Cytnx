#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
using namespace std;
int main() {
  Tensor A({3, 4, 5});

  // move the tensor to different device by creating a clone object
  Tensor B = A.to(Device.cuda + 0);
  cout << B.device_str() << endl;
  cout << A.device_str() << endl;

  return 0;
}
