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

  Storage B = A;
  Storage C = A.clone();

  cout << (B == A) << endl;  // true (share same instance)
  cout << is(B, A) << endl;  // true (share same instance)

  cout << (C == A) << endl;  // true (the same content.)
  cout << is(C, A) << endl;  // false (not share same instance)

  return 0;
}
