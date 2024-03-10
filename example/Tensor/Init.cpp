#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
using namespace std;
int main() {
  /*
      1. Create a Tensor with
      shape (3,4,5),
      dtype =Type.Double [default],
      device=Device.cpu [default]
  */
  Tensor A({3, 4, 5});
  cout << A << endl;

  /*
      2. Create a Tensor with
      shape (3,4,5),
      dtype =Type.Uint64,
      device=Device.cpu [default],
      [Note] the dtype can be any one of the supported type.
  */
  Tensor B({3, 4, 5}, Type.Uint64);
  cout << B << endl;

  /*
      3. Initialize a Tensor with
      shape (3,4,5),
      dtype =Type.Double,
      device=Device.cuda+0, (on gpu with gpu-id=0)
      [Note] the gpu device can be set with Device.cuda+<gpu-id>
  */
  Tensor C({3, 4, 5}, Type.Double, Device.cuda + 0);
  cout << C << endl;

  // 4. Create an empty Tensor, and init later
  Tensor D;
  D.Init({3, 4, 5}, Type.Double, Device.cpu);

  return 0;
}
