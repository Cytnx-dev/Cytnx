#include "cytnx.hpp"

int main() {
  cytnx::Tensor tensor({1});
  return tensor.shape()[0] == 1 ? 0 : 1;
}
