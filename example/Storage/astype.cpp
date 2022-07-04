#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
using namespace std;
int main() {
  /*
      1. Create a Storage with
      dtype =Type.Double [default],
  */
  Storage A(10);
  cout << A.dtype_str() << endl;

  // the type is the same as A's type, so B shares the same object with A
  Storage B = A.astype(Type.Double);
  cout << B.dtype_str() << endl;
  cout << is(B, A) << endl;  // true

  // cast A from Type.Double to Type.Float
  Storage C = A.astype(Type.Float);
  cout << C.dtype_str() << endl;
  cout << is(C, A) << endl;  // false

  Storage D(10, Type.Double, Device.cuda + 0);
  // D is on GPU, so E is also on GPU
  Storage E = D.astype(Type.Float);

  cout << E.device_str() << endl;

  return 0;
}
