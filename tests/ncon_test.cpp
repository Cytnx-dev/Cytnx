#include "ncon_test.h"

using namespace std;
using namespace cytnx;

TEST_F(NconTest, ncon) {
  double dans;
  ifstream fin("answer.txt");
  UniTensor res = ncon(input.first, input.second, true);
  res.reshape_({-1});

  for (int i = 0; i < res.shape()[0]; i++) {
    fin >> dans;
    double dres = *(((double*)res.get_block_().storage().data()) + i);
    EXPECT_NEAR(dans, dres, 1e-8);  // We should consider change the epsilon since the resulting
                                    // tensor may be very large in number
  }
}
