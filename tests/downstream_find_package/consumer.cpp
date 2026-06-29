// Downstream consumer smoke test. Exercises a code path that runs through
// HPTT (Tensor::permute dispatches to the HPTT-backed tensor transpose in
// Movemem_cpu.cpp when cytnx is built with USE_HPTT=ON), so a clean link
// and a correct result together prove the exported package wired up both
// cytnx itself and its bundled HPTT/OpenMP dependencies.
#include <cytnx.hpp>

#include <iostream>

int main() {
  using namespace cytnx;

  // A 2x3 tensor with values 0..5, then permute the two axes. permute goes
  // through the HPTT path; if libhptt or OpenMP failed to link, this would
  // not resolve at link time.
  Tensor t = arange(6).reshape(2, 3);
  Tensor tp = t.permute(1, 0).contiguous();

  if (tp.shape().size() != 2 || tp.shape()[0] != 3 || tp.shape()[1] != 2) {
    std::cerr << "FAIL: unexpected permuted shape\n";
    return 1;
  }

  // t[i, j] == tp[j, i]; check one off-diagonal element to confirm the
  // transpose actually moved data rather than returning a no-op.
  if (tp.at<double>(2, 1) != t.at<double>(1, 2)) {
    std::cerr << "FAIL: permuted element mismatch\n";
    return 1;
  }

  std::cout << "OK: downstream find_package(Cytnx) consumer linked and ran\n";
  return 0;
}
