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
  cout << B.is_contiguous() << endl;  // false.
  B.contiguous_();  // actual moving the memory
  cout << B.is_contiguous() << endl;  // true.

  return 0;
}
