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
  Tensor C = B.contiguous();  // actual moving the memory
  std::cout << B.is_contiguous() << std::endl;  // false.
  std::cout << C.is_contiguous() << std::endl;  // true.
  std::cout << C.shape() << std::endl;

  std::cout << C.same_data(B) << std::endl;  // false
  std::cout << B.same_data(A) << std::endl;  // true

  return 0;
}
