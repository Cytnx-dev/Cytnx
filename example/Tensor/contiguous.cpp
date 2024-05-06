#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
using namespace std;
int main() {
  Tensor A({3, 4, 5});
  cout << A.shape() << endl;

  Tensor B = A.permute({0, 2, 1});
  cout << B.shape() << endl;

  //[Note] permute will not actually move the internal memory (storage) layout.
  //       this is called non-contiguous status.
  //       the memory layout will only move when Tensor.contiguous() is called.
  Tensor C = B.contiguous();  // actual moving the memory
  cout << B.is_contiguous() << endl;  // false.
  cout << C.is_contiguous() << endl;  // true.
  cout << C.shape() << endl;

  cout << C.same_data(B) << endl;  // false
  cout << B.same_data(A) << endl;  // true

  return 0;
}
