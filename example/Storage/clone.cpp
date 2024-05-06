#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
using namespace std;
int main() {
  Storage A(15);

  Storage B = A;  // B shares same object with A
  Storage C = A.clone();  // C is a copy of A

  // use is() to check if two variable shares same object
  cout << is(B, A) << endl;
  cout << is(C, A) << endl;

  return 0;
}
