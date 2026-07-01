#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
int main() {
  Tensor A({3, 4, 5});

  // move the tensor to different device by creating a clone object
  Tensor B = A.to(Device.cuda + 0);
  std::cout << B.device_str() << std::endl;
  std::cout << A.device_str() << std::endl;

  return 0;
}
