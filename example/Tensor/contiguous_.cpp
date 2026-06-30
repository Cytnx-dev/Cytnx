#include "cytnx.hpp"
#include <iostream>

using namespace cytnx;
int main() {
  Tensor A({3, 4, 5});
  std::cout << A.shape() << std::endl;

  Tensor B = A.permute({0, 2, 1});
  std::cout << B.shape() << std::endl;

  //[Note] permute will not actually move the internal memory (storage) layout.
  //       this is called non-contiguous status.
  //       the memory layout will only move when Tensor.contiguous() is called.
  std::cout << B.is_contiguous() << std::endl;  // false.
  B.contiguous_();  // actual moving the memory
  std::cout << B.is_contiguous() << std::endl;  // true.

  return 0;
}
