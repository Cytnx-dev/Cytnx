#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
using namespace std;
int main() {
  /*
      Get the real part from a complex128 (ComplexDouble) Storage
  */
  Storage S1(10, Type.ComplexDouble);
  for (unsigned int i = 0; i < S1.size(); i++) {
    S1.at<cytnx_complex128>(i) = cytnx_complex128(i, i + 1);
  }
  cout << S1 << endl;

  Storage S1r = S1.real();
  cout << S1r << endl;

  /*
      Get the real part from a complex64 (ComplexFloat) Storage
  */
  Storage S2(10, Type.ComplexFloat);
  for (unsigned int i = 0; i < S1.size(); i++) {
    S2.at<cytnx_complex64>(i) = cytnx_complex64(i, i + 2);
  }
  cout << S2 << endl;

  Storage S2r = S2.real();
  cout << S2r << endl;

  return 0;
}
