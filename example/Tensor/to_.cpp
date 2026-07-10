#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
int main() {
  Tensor A({3, 4, 5});

  // move the instance tensor to different device
  A.to_(Device.cuda + 0);
  std::cout << A.device_str() << std::endl;

  return 0;
}
