#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
int main() {
  /*
      Get the real part from a complex128 (ComplexDouble) Storage
  */
  Storage S1(10, Type.ComplexDouble);
  for (unsigned int i = 0; i < S1.size(); i++) {
    S1.at<cytnx_complex128>(i) = cytnx_complex128(i, i + 1);
  }
  std::cout << S1 << std::endl;

  Storage S1r = S1.real();
  std::cout << S1r << std::endl;

  /*
      Get the real part from a complex64 (ComplexFloat) Storage
  */
  Storage S2(10, Type.ComplexFloat);
  for (unsigned int i = 0; i < S1.size(); i++) {
    S2.at<cytnx_complex64>(i) = cytnx_complex64(i, i + 2);
  }
  std::cout << S2 << std::endl;

  Storage S2r = S2.real();
  std::cout << S2r << std::endl;

  return 0;
}
