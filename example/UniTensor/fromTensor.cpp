#include "cytnx.hpp"
#include <iostream>

using namespace std;
using namespace cytnx;

int main(int argc, char* argv[]) {
  Tensor T = zeros({4, 4});
  UniTensor CyT = UniTensor(T, false, 2);  // create un-tagged CyTensor from Tensor
  CyT.print_diagram();

  cout << "before:" << endl;
  cout << T << endl;
  cout << CyT << endl;

  // Note that it is a shared view, so a change to CyT will affect Tensor T.
  CyT.at<cytnx_double>({0, 0}) = 1.456;

  cout << "after:" << endl;
  cout << T << endl;
  cout << CyT << endl;

  // If we want a new instance of memery, use clone at initialize:
  cout << "[non-share example]" << endl;
  UniTensor CyT_nonshare = UniTensor(T.clone(), false, 2);

  cout << "before:" << endl;
  cout << T << endl;
  cout << CyT_nonshare << endl;

  CyT_nonshare.at<cytnx_double>({1, 1}) = 2.345;

  cout << "after" << endl;
  cout << T << endl;  // T is unchanged!
  cout << CyT_nonshare << endl;

  return 0;
}
