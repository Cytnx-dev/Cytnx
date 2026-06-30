#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;

int main(int argc, char* argv[]) {
  Tensor T = zeros({4, 4});
  UniTensor CyT = UniTensor(T, false, 2);  // create un-tagged CyTensor from Tensor
  CyT.print_diagram();

  std::cout << "before:" << std::endl;
  std::cout << T << std::endl;
  std::cout << CyT << std::endl;

  // Note that it is a shared view, so a change to CyT will affect Tensor T.
  CyT.at<cytnx_double>({0, 0}) = 1.456;

  std::cout << "after:" << std::endl;
  std::cout << T << std::endl;
  std::cout << CyT << std::endl;

  // If we want a new instance of memery, use clone at initialize:
  std::cout << "[non-share example]" << std::endl;
  UniTensor CyT_nonshare = UniTensor(T.clone(), false, 2);

  std::cout << "before:" << std::endl;
  std::cout << T << std::endl;
  std::cout << CyT_nonshare << std::endl;

  CyT_nonshare.at<cytnx_double>({1, 1}) = 2.345;

  std::cout << "after" << std::endl;
  std::cout << T << std::endl;  // T is unchanged!
  std::cout << CyT_nonshare << std::endl;

  return 0;
}
