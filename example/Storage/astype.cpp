#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
int main() {
  /*
      1. Create a Storage with
      dtype =Type.Double [default],
  */
  Storage A(10);
  std::cout << A.dtype_str() << std::endl;

  // the type is the same as A's type, so B shares the same object with A
  Storage B = A.astype(Type.Double);
  std::cout << B.dtype_str() << std::endl;
  std::cout << is(B, A) << std::endl;  // true

  // cast A from Type.Double to Type.Float
  Storage C = A.astype(Type.Float);
  std::cout << C.dtype_str() << std::endl;
  std::cout << is(C, A) << std::endl;  // false

  Storage D(10, Type.Double, Device.cuda + 0);
  // D is on GPU, so E is also on GPU
  Storage E = D.astype(Type.Float);

  std::cout << E.device_str() << std::endl;

  return 0;
}
