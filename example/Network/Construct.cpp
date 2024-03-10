#include "cytnx.hpp"
#include <iostream>
using namespace cytnx;
using namespace std;

int main(int argc, char* argv[]) {
  auto ta = UniTensor(arange(24).reshape(2, 3, 4), 1);
  auto tb = UniTensor(arange(24).reshape(2, 3, 4), 1);
  auto tc = UniTensor(arange(24).reshape(2, 3, 4), 1);
  auto td = UniTensor(arange(24).reshape(2, 3, 4), 1);

  ta.set_labels({0, 1, 2});
  tb.set_labels({0, 3, 4});
  tc.set_labels({5, 1, 6});
  td.set_labels({5, 7, 8});

  UniTensor oot =
    Network::Contract(
      {ata, atb, atc, atd},  // input tensors
      "2,3,4;6,7,8",  // output tensor label ordering and rowrank
      {},  // is clone mask. default all clone if empty
      {"A", "B", "C", "D"},  // input tensor alias (only needed if manually assign order
      "(A,B),(C,D)")  // contraction order [optional]
      .Launch();
}
