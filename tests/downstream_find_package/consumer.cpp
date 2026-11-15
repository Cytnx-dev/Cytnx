// Downstream consumer smoke test. Materialize a permuted tensor large enough
// to enter HPTT, checking both exported dependency linkage and execution.
#include <cytnx.hpp>

#include <iostream>

int main() {
  // More than 64 elements are needed to enter Cytnx's HPTT transpose path.
  const cytnx::Tensor source = cytnx::arange(99).reshape(9, 11);
  const cytnx::Tensor result = source.permute(1, 0).contiguous();

  if (result.shape().size() != 2 || result.shape()[0] != 11 || result.shape()[1] != 9 ||
      result.dtype() != cytnx::Type.Double) {
    std::cerr << "FAIL: unexpected permuted shape or dtype\n";
    return 1;
  }

  // Independent row-major oracle for every coefficient of the transposed tensor.
  for (int i = 0; i < 11; ++i) {
    for (int j = 0; j < 9; ++j) {
      if (result.at<double>(i, j) != j * 11 + i) {
        std::cerr << "FAIL: permuted element mismatch\n";
        return 1;
      }
    }
  }

  std::cout << "OK: downstream find_package(Cytnx) consumer linked and ran\n";
  return 0;
}
