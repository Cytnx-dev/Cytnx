#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
using namespace std;
int main() {
  /*
      1. Create a Storage with
      10 elements,
      dtype =Type.Double [default],
      device=Device.cpu [default]
  */
  Storage A(10);
  cout << A << endl;

  /*
      2. Create a Storage with
      10 elements,
      dtype =Type.Uint64,
      device=Device.cpu [default],
      [Note] the dtype can be any one of the supported type.
  */
  Storage B(10, Type.Uint64);
  cout << B << endl;

  /*
      3. Initialize a Storage with
      10 elements,
      dtype =Type.Double,
      device=Device.cuda+0, (on gpu with gpu-id=0)
      [Note] the gpu device can be set with Device.cuda+<gpu-id>
  */
  Storage C(10, Type.Double, Device.cuda + 0);
  cout << C << endl;

  // 4. Create an empty Storage, and init later
  Storage D;
  D.Init(10, Type.Double, Device.cpu);

  return 0;
}
