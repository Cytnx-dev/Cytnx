#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
using namespace std;
int main() {
  Tensor A = arange(30, Type.Float).reshape(2, 3, 5);
  cout << A << endl;

  // note that type resolver should be consist with the dtype
  cout << A.at<cytnx_float>(0, 0, 2) << endl;

  // the return is a ref., can be modify directly.
  A.at<cytnx_float>(0, 0, 2) = 999;

  cout << A.at<cytnx_float>(0, 0, 2) << endl;

  // [Note] there are two way to give argument:
  // Method 1: more like 'c++' way:
  //           (alternatively, you can also simply give a std::vector)
  A.at<cytnx_float>({0, 0, 2});  // note the braket{}

  // Method 2: more like 'python' way:
  A.at<cytnx_float>(0, 0, 2);

  return 0;
}
